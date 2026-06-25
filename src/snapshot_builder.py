"""跨层多池快照构建：Multicall3 钉块采集 + JSON 序列化（Phase 1）。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from web3 import Web3

from config.settings import COST_DEFAULTS, INVENTORY_DEFAULT, RPC
from src.multicall_utils import (
    aggregate3_call,
    decode_liquidity,
    decode_slot0,
    decode_tick,
    encode_contract_call,
)
from src.pool_registry import PoolRegistryEntry, get_contract_addresses, load_registry
from src.pool_utils import POOL_ABI, sqrt_price_x96_to_eth_usdc_meta, tick_spacing_for_fee
from src.rpc_utils import create_sync_w3
from src.types import GasState, InventoryState, MultiPoolSnapshot, PoolSnapshot

logger = logging.getLogger(__name__)

ARBSYS_ADDRESS = "0x0000000000000000000000000000000000000064"
NODE_INTERFACE_ADDRESS = "0x00000000000000000000000000000000000000c8"
ARBSYS_ABI = [
    {
        "inputs": [],
        "name": "getL1BlockNumber",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getL1BatchNumber",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]
NODE_INTERFACE_ABI = [
    {
        "inputs": [{"name": "l2BlockNum", "type": "uint64"}],
        "name": "blockL1Num",
        "outputs": [{"name": "l1BlockNum", "type": "uint64"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "block", "type": "uint64"}],
        "name": "findBatchContainingBlock",
        "outputs": [{"name": "batch", "type": "uint64"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _block_timestamp_iso(w3: Web3, block_number: int) -> str:
    block = w3.eth.get_block(block_number)
    ts = int(block["timestamp"])
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gas_gwei_at_block(w3: Web3, block_number: int, priority_gwei: float = 2.0) -> float:
    block = w3.eth.get_block(block_number)
    base_fee = block.get("baseFeePerGas")
    if base_fee is None:
        try:
            return float(w3.from_wei(w3.eth.gas_price, "gwei"))
        except Exception:
            return priority_gwei
    return float(base_fee) / 1e9 + priority_gwei


def _l1_block_from_mix_hash(block: dict) -> int | None:
    """Arbitrum Nitro：mixHash 第 2～8 字节为 l1BlockNumber（部分 RPC 不单独返回该字段）。"""
    mix = block.get("mixHash")
    if mix is None:
        return None
    if isinstance(mix, str):
        raw = bytes.fromhex(mix.removeprefix("0x"))
    elif isinstance(mix, (bytes, bytearray)):
        raw = bytes(mix)
    else:
        return None
    if len(raw) < 16:
        return None
    value = int.from_bytes(raw[8:16], "big")
    return value if value > 0 else None


def resolve_l1_batch(w3_arb: Web3, l2_block: int) -> int | None:
    """Arbitrum L2 块 → L1 祖先块号（存入 snapshot.l1_batch，MVP 对齐字段）。

    尝试顺序：RPC l1BlockNumber → mixHash 解码 → ArbSys.getL1BlockNumber
    → NodeInterface.blockL1Num → NodeInterface.findBatchContainingBlock。
    """
    try:
        block = w3_arb.eth.get_block(l2_block)
        l1_block_number = block.get("l1BlockNumber")
        if l1_block_number is not None:
            return int(l1_block_number)
        from_mix = _l1_block_from_mix_hash(block)
        if from_mix is not None:
            return from_mix
    except Exception as exc:
        logger.debug("l1BlockNumber/mixHash unavailable on block %s: %s", l2_block, exc)

    try:
        arb_sys = w3_arb.eth.contract(
            address=Web3.to_checksum_address(ARBSYS_ADDRESS),
            abi=ARBSYS_ABI,
        )
        return int(arb_sys.functions.getL1BlockNumber().call(block_identifier=l2_block))
    except Exception as exc:
        logger.debug("ArbSys getL1BlockNumber failed at %s: %s", l2_block, exc)

    try:
        node_iface = w3_arb.eth.contract(
            address=Web3.to_checksum_address(NODE_INTERFACE_ADDRESS),
            abi=NODE_INTERFACE_ABI,
        )
        return int(
            node_iface.functions.blockL1Num(l2_block).call(block_identifier=l2_block)
        )
    except Exception as exc:
        logger.debug("NodeInterface blockL1Num failed at %s: %s", l2_block, exc)

    try:
        node_iface = w3_arb.eth.contract(
            address=Web3.to_checksum_address(NODE_INTERFACE_ADDRESS),
            abi=NODE_INTERFACE_ABI,
        )
        return int(
            node_iface.functions.findBatchContainingBlock(l2_block).call(
                block_identifier=l2_block
            )
        )
    except Exception as exc:
        logger.debug("NodeInterface findBatchContainingBlock failed at %s: %s", l2_block, exc)

    return None


def _fetch_slot0_liquidity_batch(
    w3: Web3,
    multicall_address: str,
    pools: list[PoolRegistryEntry],
    block_number: int,
) -> list[tuple[int, int, int]]:
    """批量拉 slot0 + liquidity，返回 (sqrt_price_x96, tick, liquidity)。"""
    calls: list[tuple[str, bytes]] = []
    for pool in pools:
        addr = pool.checksum_address()
        contract = w3.eth.contract(address=addr, abi=POOL_ABI)
        calls.append((addr, encode_contract_call(contract.functions.slot0())))
        calls.append((addr, encode_contract_call(contract.functions.liquidity())))

    results = aggregate3_call(w3, multicall_address, calls, block_number)
    if len(results) != len(calls):
        raise RuntimeError(f"Multicall result count mismatch: {len(results)} != {len(calls)}")

    parsed: list[tuple[int, int, int]] = []
    for i in range(0, len(results), 2):
        ok0, data0 = results[i]
        ok1, data1 = results[i + 1]
        if not ok0 or not ok1:
            raise RuntimeError(f"Multicall failed for pool index {i // 2}")
        sqrt_price, tick = decode_slot0(data0)
        liquidity = decode_liquidity(data1)
        parsed.append((sqrt_price, tick, liquidity))
    return parsed


def _fetch_primary_ticks(
    w3: Web3,
    multicall_address: str,
    pool: PoolRegistryEntry,
    center_tick: int,
    block_number: int,
) -> list[dict[str, Any]]:
    spacing = tick_spacing_for_fee(pool.fee_tier)
    tick_values = [center_tick - spacing, center_tick, center_tick + spacing]
    addr = pool.checksum_address()
    contract = w3.eth.contract(address=addr, abi=POOL_ABI)
    calls = [
        (addr, encode_contract_call(contract.functions.ticks(t)))
        for t in tick_values
    ]
    results = aggregate3_call(w3, multicall_address, calls, block_number)
    ticks: list[dict[str, Any]] = []
    for tick_val, (ok, data) in zip(tick_values, results):
        entry: dict[str, Any] = {"tick": tick_val, "initialized": False}
        if ok and data:
            decoded = decode_tick(data)
            entry.update(decoded)
        ticks.append(entry)
    return ticks


def fetch_chain_pool_snapshots(
    w3: Web3,
    chain: str,
    block_number: int,
    *,
    fetch_primary_ticks: bool = True,
) -> list[PoolSnapshot]:
    pools = load_registry(chain)
    contracts = get_contract_addresses(chain)
    multicall = contracts.get("multicall3", "")
    if not multicall:
        raise ValueError(f"Multicall3 address missing for chain {chain}")

    block_time = _block_timestamp_iso(w3, block_number)
    slot_data = _fetch_slot0_liquidity_batch(w3, multicall, pools, block_number)

    snapshots: list[PoolSnapshot] = []
    primary_entry = next((p for p in pools if p.is_primary), pools[0])

    for pool, (sqrt_price, tick, liquidity) in zip(pools, slot_data):
        price = sqrt_price_x96_to_eth_usdc_meta(
            sqrt_price,
            pool.token0_decimals,
            pool.token1_decimals,
            pool.weth_is_token0,
        )
        ticks_loaded = False
        initialized_ticks: list[dict[str, Any]] = []

        if fetch_primary_ticks and pool.id == primary_entry.id:
            try:
                initialized_ticks = _fetch_primary_ticks(
                    w3, multicall, pool, tick, block_number
                )
                ticks_loaded = any(t.get("initialized") for t in initialized_ticks)
            except Exception as exc:
                logger.warning("Primary tick fetch failed for %s: %s", pool.id, exc)

        snap = PoolSnapshot(
            chain=chain,
            pool_address=pool.pool_address,
            token0=pool.token0,
            token1=pool.token1,
            fee_tier=pool.fee_tier,
            sqrt_price_x96=sqrt_price,
            liquidity=liquidity,
            tick=tick,
            eth_usdc_price=price,
            ticks_loaded=ticks_loaded,
            pool_id=pool.id,
            block_number=block_number,
            block_time=block_time,
            initialized_ticks=initialized_ticks or None,
        )
        snapshots.append(snap)

    return snapshots


def build_multi_pool_snapshot(
    l1_block: int,
    l2_block: int,
    *,
    snapshot_id: str | None = None,
    w3_l1: Web3 | None = None,
    w3_l2: Web3 | None = None,
    trigger: str = "manual_fetch",
    bridge_fee_usd: float | None = None,
    inventory: InventoryState | None = None,
    fetch_primary_ticks: bool = True,
) -> MultiPoolSnapshot:
    w3_l1 = w3_l1 or create_sync_w3(RPC["ethereum"]["http"])
    w3_l2 = w3_l2 or create_sync_w3(RPC["arbitrum"]["http"])

    l1_pools = fetch_chain_pool_snapshots(
        w3_l1, "ethereum", l1_block, fetch_primary_ticks=fetch_primary_ticks
    )
    l2_pools = fetch_chain_pool_snapshots(
        w3_l2, "arbitrum", l2_block, fetch_primary_ticks=fetch_primary_ticks
    )

    l1_batch = resolve_l1_batch(w3_l2, l2_block)
    l1_ts = int(w3_l1.eth.get_block(l1_block)["timestamp"])
    l2_ts = int(w3_l2.eth.get_block(l2_block)["timestamp"])
    l2_lag_ms = max(0, (l2_ts - l1_ts) * 1000)

    sid = snapshot_id or f"snap_l1{l1_block}_l2{l2_block}"
    return MultiPoolSnapshot(
        snapshot_id=sid,
        l1_block=l1_block,
        l2_block=l2_block,
        l1_pools=l1_pools,
        l2_pools=l2_pools,
        gas=GasState(
            l1_gwei=gas_gwei_at_block(w3_l1, l1_block),
            l2_gwei=gas_gwei_at_block(w3_l2, l2_block, priority_gwei=0.1),
        ),
        bridge_fee_usd=bridge_fee_usd if bridge_fee_usd is not None else COST_DEFAULTS["bridge_fee_usd"],
        inventory=inventory
        or InventoryState(
            l1_eth=INVENTORY_DEFAULT["l1_eth"],
            l2_eth=INVENTORY_DEFAULT["l2_eth"],
        ),
        trigger=trigger,
        timestamp=datetime.fromtimestamp(l1_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        l1_batch=l1_batch,
        l2_lag_ms=l2_lag_ms,
    )


def rebuild_snapshot_with_l1_block(
    base: MultiPoolSnapshot,
    l1_block: int,
    *,
    w3_l1: Web3 | None = None,
    w3_l2: Web3 | None = None,
    fetch_primary_ticks: bool = False,
) -> MultiPoolSnapshot:
    """在固定 L2 钉块下，用新的 L1 块重拉池状态（±1 batch 敏感性）。"""
    w3_l1 = w3_l1 or create_sync_w3(RPC["ethereum"]["http"])
    w3_l2 = w3_l2 or create_sync_w3(RPC["arbitrum"]["http"])

    l1_pools = fetch_chain_pool_snapshots(
        w3_l1, "ethereum", l1_block, fetch_primary_ticks=fetch_primary_ticks
    )
    l1_ts = int(w3_l1.eth.get_block(l1_block)["timestamp"])
    l2_ts = int(w3_l2.eth.get_block(base.l2_block)["timestamp"])
    l2_lag_ms = max(0, (l2_ts - l1_ts) * 1000)

    return MultiPoolSnapshot(
        snapshot_id=base.snapshot_id,
        l1_block=l1_block,
        l2_block=base.l2_block,
        l1_pools=l1_pools,
        l2_pools=base.l2_pools,
        gas=GasState(
            l1_gwei=gas_gwei_at_block(w3_l1, l1_block),
            l2_gwei=base.gas.l2_gwei,
        ),
        bridge_fee_usd=base.bridge_fee_usd,
        inventory=base.inventory,
        trigger="batch_sensitivity",
        timestamp=datetime.fromtimestamp(l1_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        l1_batch=base.l1_batch,
        l2_lag_ms=l2_lag_ms,
        swap_amount_usd=base.swap_amount_usd,
        scenario=base.scenario,
    )


def _pool_snapshot_to_dict(pool: PoolSnapshot) -> dict[str, Any]:
    data: dict[str, Any] = {
        "pool_id": pool.pool_id,
        "chain": pool.chain,
        "pool_address": pool.pool_address,
        "token0": pool.token0,
        "token1": pool.token1,
        "fee_tier": pool.fee_tier,
        "sqrt_price_x96": str(pool.sqrt_price_x96),
        "liquidity": str(pool.liquidity),
        "tick": pool.tick,
        "eth_usdc_price": pool.eth_usdc_price,
        "ticks_loaded": pool.ticks_loaded,
        "block_number": pool.block_number,
        "block_time": pool.block_time,
    }
    extra_ticks = pool.initialized_ticks
    if extra_ticks:
        data["initialized_ticks"] = extra_ticks
    return data


def multi_pool_snapshot_to_dict(snapshot: MultiPoolSnapshot) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "l1_block": snapshot.l1_block,
        "l2_block": snapshot.l2_block,
        "l1_batch": snapshot.l1_batch,
        "timestamp": snapshot.timestamp,
        "trigger": snapshot.trigger,
        "l2_lag_ms": snapshot.l2_lag_ms,
        "swap_amount_usd": snapshot.swap_amount_usd,
        "scenario": snapshot.scenario,
        "gas": {"l1_gwei": snapshot.gas.l1_gwei, "l2_gwei": snapshot.gas.l2_gwei},
        "bridge_fee_usd": snapshot.bridge_fee_usd,
        "inventory": {
            "l1_eth": snapshot.inventory.l1_eth,
            "l2_eth": snapshot.inventory.l2_eth,
        },
        "l1_pools": [_pool_snapshot_to_dict(p) for p in snapshot.l1_pools],
        "l2_pools": [_pool_snapshot_to_dict(p) for p in snapshot.l2_pools],
    }


def multi_pool_snapshot_from_dict(data: dict[str, Any]) -> MultiPoolSnapshot:
    def _parse_pool(item: dict[str, Any]) -> PoolSnapshot:
        pool = PoolSnapshot(
            chain=item["chain"],
            pool_address=item["pool_address"],
            token0=item["token0"],
            token1=item["token1"],
            fee_tier=int(item["fee_tier"]),
            sqrt_price_x96=int(item["sqrt_price_x96"]),
            liquidity=int(item["liquidity"]),
            tick=int(item["tick"]),
            eth_usdc_price=float(item["eth_usdc_price"]),
            ticks_loaded=bool(item.get("ticks_loaded", False)),
            pool_id=item.get("pool_id", ""),
            block_number=int(item.get("block_number", 0)),
            block_time=item.get("block_time", ""),
            initialized_ticks=item.get("initialized_ticks"),
        )
        return pool

    inv = data.get("inventory", INVENTORY_DEFAULT)
    gas = data.get("gas", {})
    return MultiPoolSnapshot(
        snapshot_id=data["snapshot_id"],
        l1_block=int(data["l1_block"]),
        l2_block=int(data["l2_block"]),
        l1_pools=[_parse_pool(p) for p in data["l1_pools"]],
        l2_pools=[_parse_pool(p) for p in data["l2_pools"]],
        gas=GasState(
            l1_gwei=float(gas.get("l1_gwei", 20.0)),
            l2_gwei=float(gas.get("l2_gwei", 0.1)),
        ),
        bridge_fee_usd=float(data.get("bridge_fee_usd", COST_DEFAULTS["bridge_fee_usd"])),
        inventory=InventoryState(
            l1_eth=float(inv.get("l1_eth", INVENTORY_DEFAULT["l1_eth"])),
            l2_eth=float(inv.get("l2_eth", INVENTORY_DEFAULT["l2_eth"])),
        ),
        trigger=data.get("trigger", "loaded"),
        timestamp=data.get("timestamp", ""),
        l1_batch=data.get("l1_batch"),
        l2_lag_ms=int(data.get("l2_lag_ms", 0)),
        swap_amount_usd=data.get("swap_amount_usd"),
        scenario=data.get("scenario"),
    )


def save_multi_pool_snapshot(snapshot: MultiPoolSnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = multi_pool_snapshot_to_dict(snapshot)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_multi_pool_snapshot(path: Path) -> MultiPoolSnapshot:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return multi_pool_snapshot_from_dict(data)
