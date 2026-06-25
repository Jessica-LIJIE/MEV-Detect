"""Phase 1：多池快照 JSON 序列化与 Multicall 解码测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from eth_abi import encode

from src.multicall_utils import decode_liquidity, decode_slot0, decode_tick
from src.snapshot_builder import (
    multi_pool_snapshot_from_dict,
    multi_pool_snapshot_to_dict,
    resolve_l1_batch,
)
from src.types import GasState, InventoryState, MultiPoolSnapshot, PoolSnapshot


def _sample_snapshot() -> MultiPoolSnapshot:
    pool_l1 = PoolSnapshot(
        chain="ethereum",
        pool_address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        token0="USDC",
        token1="WETH",
        fee_tier=500,
        sqrt_price_x96=79350000000000000000000000000,
        liquidity=1850000000000000000000,
        tick=201350,
        eth_usdc_price=3510.5,
        ticks_loaded=True,
        pool_id="eth_usdc_005",
        block_number=20100100,
        block_time="2025-06-01T08:15:30Z",
        initialized_ticks=[
            {"tick": 201340, "liquidityNet": "1000", "initialized": True},
            {"tick": 201350, "liquidityNet": "2000", "initialized": True},
            {"tick": 201360, "liquidityNet": "-500", "initialized": True},
        ],
    )
    pool_l2 = PoolSnapshot(
        chain="arbitrum",
        pool_address="0xC31E54c7a869B9Fc0ccE9654dD8772aB00ec6F36",
        token0="WETH",
        token1="USDC",
        fee_tier=500,
        sqrt_price_x96=79080000000000000000000000000,
        liquidity=920000000000000000000,
        tick=201080,
        eth_usdc_price=3505.8,
        ticks_loaded=False,
        pool_id="arb_usdc_005",
        block_number=220001500,
        block_time="2025-06-01T08:15:31Z",
    )
    return MultiPoolSnapshot(
        snapshot_id="snap_test_001",
        l1_block=20100100,
        l2_block=220001500,
        l1_pools=[pool_l1] * 6,
        l2_pools=[pool_l2] * 6,
        gas=GasState(l1_gwei=22.0, l2_gwei=0.1),
        bridge_fee_usd=5.0,
        inventory=InventoryState(l1_eth=50.0, l2_eth=50.0),
        trigger="test",
        timestamp="2025-06-01T08:15:30Z",
        l1_batch=20099999,
        l2_lag_ms=980,
    )


def test_multicall_decoders():
    slot0_bytes = encode(
        ["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
        [79350000000000000000000000000, 201350, 0, 0, 0, 0, True],
    )
    sqrt_price, tick = decode_slot0(slot0_bytes)
    assert tick == 201350
    assert sqrt_price > 0

    liq_bytes = encode(["uint128"], [1850000000000000000000])
    assert decode_liquidity(liq_bytes) == 1850000000000000000000

    tick_bytes = encode(
        ["uint128", "int128", "uint256", "uint256", "int56", "uint160", "uint32", "bool"],
        [1000, 2000, 0, 0, 0, 0, 0, True],
    )
    decoded = decode_tick(tick_bytes)
    assert decoded["initialized"] is True
    assert decoded["liquidityNet"] == "2000"


def test_snapshot_json_roundtrip():
    original = _sample_snapshot()
    data = multi_pool_snapshot_to_dict(original)
    assert len(data["l1_pools"]) == 6
    assert len(data["l2_pools"]) == 6
    assert data["l1_batch"] == 20099999
    assert data["l1_pools"][0]["initialized_ticks"]

    restored = multi_pool_snapshot_from_dict(data)
    assert restored.snapshot_id == original.snapshot_id
    assert restored.num_l1_pools == 6
    assert restored.num_l2_pools == 6
    assert restored.l1_pools[0].eth_usdc_price == pytest.approx(3510.5)
    assert restored.l1_pools[0].initialized_ticks is not None
    assert len(restored.l1_pools[0].initialized_ticks) == 3


def test_resolve_l1_batch_from_rpc_field():
    w3 = MagicMock()
    w3.eth.get_block.return_value = {"l1BlockNumber": 20099999, "timestamp": 0}
    assert resolve_l1_batch(w3, 220001500) == 20099999


def test_resolve_l1_batch_from_mix_hash():
    w3 = MagicMock()
    mix_bytes = b"\x00" * 8 + (20099999).to_bytes(8, "big") + b"\x00" * 16
    w3.eth.get_block.return_value = {"mixHash": "0x" + mix_bytes.hex(), "timestamp": 0}
    assert resolve_l1_batch(w3, 220001500) == 20099999


def test_resolve_l1_batch_returns_none_on_failure():
    w3 = MagicMock()
    w3.eth.get_block.side_effect = RuntimeError("no block")
    w3.eth.contract.side_effect = RuntimeError("no arb sys")
    assert resolve_l1_batch(w3, 1) is None
