"""Mock 多池快照加载与构建（Phase 4 消融，与 live 数据分目录）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import MOCK_MULTIPOOL_DIR
from src.pool_registry import PoolRegistryEntry, load_registry
from src.types import GasState, InventoryState, MultiPoolSnapshot, PoolSnapshot

DEFAULT_MOCK_MULTIPOOL_PATH = MOCK_MULTIPOOL_DIR / "records.json"
DEFAULT_LIQUIDITY = 3_000_000_000_000_000_000


@dataclass(frozen=True)
class MockMultipoolRecord:
    """紧凑 Mock 记录 + 元数据（用于消融实验标注）。"""

    snapshot: MultiPoolSnapshot
    category: str
    scenario: str
    best_pair: tuple[int, int] | None = None
    latency_risk_lambda: float | None = None


def _registry_pool_to_snapshot(
    entry: PoolRegistryEntry,
    *,
    eth_usdc_price: float,
    liquidity: int,
    block_number: int,
    block_time: str,
) -> PoolSnapshot:
    return PoolSnapshot(
        chain=entry.chain,
        pool_address=entry.pool_address,
        token0=entry.token0,
        token1=entry.token1,
        fee_tier=entry.fee_tier,
        sqrt_price_x96=1,
        liquidity=liquidity,
        tick=0,
        eth_usdc_price=eth_usdc_price,
        pool_id=entry.id,
        block_number=block_number,
        block_time=block_time,
    )


def build_multipool_snapshot_from_prices(
    record_id: str,
    l1_prices: list[float],
    l2_prices: list[float],
    *,
    scenario: str = "",
    trigger: str = "mock_multipool",
    liquidity: int = DEFAULT_LIQUIDITY,
    gas: dict[str, float] | None = None,
    inventory: dict[str, float] | None = None,
    bridge_fee_usd: float = 5.0,
    l2_lag_ms: int = 500,
    timestamp: str = "2025-06-01T12:00:00Z",
    l1_block: int = 20_100_100,
    l2_block: int = 220_001_500,
) -> MultiPoolSnapshot:
    l1_registry = load_registry("ethereum")
    l2_registry = load_registry("arbitrum")
    if len(l1_prices) != len(l1_registry) or len(l2_prices) != len(l2_registry):
        raise ValueError("l1_prices / l2_prices length must match registry pool count (6)")

    gas_data = gas or {"l1_gwei": 15.0, "l2_gwei": 0.08}
    inv = inventory or {"l1_eth": 50.0, "l2_eth": 50.0}

    return MultiPoolSnapshot(
        snapshot_id=record_id,
        l1_block=l1_block,
        l2_block=l2_block,
        l1_pools=[
            _registry_pool_to_snapshot(
                entry,
                eth_usdc_price=l1_prices[i],
                liquidity=liquidity,
                block_number=l1_block,
                block_time=timestamp,
            )
            for i, entry in enumerate(l1_registry)
        ],
        l2_pools=[
            _registry_pool_to_snapshot(
                entry,
                eth_usdc_price=l2_prices[j],
                liquidity=liquidity,
                block_number=l2_block,
                block_time=timestamp,
            )
            for j, entry in enumerate(l2_registry)
        ],
        gas=GasState(l1_gwei=float(gas_data["l1_gwei"]), l2_gwei=float(gas_data["l2_gwei"])),
        bridge_fee_usd=float(bridge_fee_usd),
        inventory=InventoryState(l1_eth=float(inv["l1_eth"]), l2_eth=float(inv["l2_eth"])),
        trigger=trigger,
        timestamp=timestamp,
        l2_lag_ms=int(l2_lag_ms),
        scenario=scenario or None,
    )


def _parse_best_pair(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    return None


def compact_record_to_mock(record: dict[str, Any]) -> MockMultipoolRecord:
    liquidity = int(record.get("liquidity", DEFAULT_LIQUIDITY))
    snapshot = build_multipool_snapshot_from_prices(
        record["id"],
        list(record["l1_prices"]),
        list(record["l2_prices"]),
        scenario=record.get("scenario", ""),
        liquidity=liquidity,
        gas=record.get("gas"),
        inventory=record.get("inventory"),
        bridge_fee_usd=float(record.get("bridge_fee_usd", 5.0)),
        l2_lag_ms=int(record.get("l2_lag_ms", 500)),
        timestamp=record.get("timestamp", "2025-06-01T12:00:00Z"),
        l1_block=int(record.get("l1_block", 20_100_100)),
        l2_block=int(record.get("l2_block", 220_001_500)),
    )
    return MockMultipoolRecord(
        snapshot=snapshot,
        category=str(record.get("category", "unknown")),
        scenario=str(record.get("scenario", "")),
        best_pair=_parse_best_pair(record.get("best_pair")),
        latency_risk_lambda=record.get("latency_risk_lambda"),
    )


def load_mock_multipool_records(
    path: Path | None = None,
) -> list[MockMultipoolRecord]:
    mock_path = path or DEFAULT_MOCK_MULTIPOOL_PATH
    with open(mock_path, encoding="utf-8") as f:
        data = json.load(f)
    return [compact_record_to_mock(item) for item in data["records"]]


def load_mock_multipool_snapshots(path: Path | None = None) -> list[MultiPoolSnapshot]:
    return [item.snapshot for item in load_mock_multipool_records(path)]
