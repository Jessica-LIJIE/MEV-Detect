"""Phase 0：池注册表与多池类型自检。"""

from pathlib import Path

import pytest

from config.settings import (
    BRIDGE_CONFIG,
    CYCLE_FINDER_CONFIG,
    INVENTORY_DEFAULT,
    POOL_REGISTRY_PATH,
)
from src.pool_registry import (
    get_contract_addresses,
    get_primary_pool,
    load_registry,
    registry_pool_count,
)
from src.types import CandidateRoute, InventoryState, MultiPoolSnapshot, PoolSnapshot


def test_registry_file_exists():
    assert POOL_REGISTRY_PATH.is_file()


def test_registry_six_pools_per_chain():
    counts = registry_pool_count()
    assert counts["ethereum"] == 6
    assert counts["arbitrum"] == 6
    assert len(load_registry("ethereum")) == 6
    assert len(load_registry("arbitrum")) == 6


def test_primary_pools():
    l1 = get_primary_pool("ethereum")
    l2 = get_primary_pool("arbitrum")
    assert l1.is_primary
    assert l2.is_primary
    assert l1.id == "eth_usdc_005"
    assert l2.id == "arb_usdc_005"


def test_token_addresses_resolved():
    eth_pools = load_registry("ethereum")
    usdc_weth = next(p for p in eth_pools if p.id == "eth_usdc_005")
    assert usdc_weth.token0 == "USDC"
    assert usdc_weth.token1 == "WETH"
    assert usdc_weth.token0_address.startswith("0x")
    assert usdc_weth.token1_address.startswith("0x")
    assert usdc_weth.weth_is_token0 is False


def test_contract_addresses():
    addrs = get_contract_addresses("ethereum")
    assert addrs["quoter_v2"].startswith("0x")
    assert addrs["multicall3"].startswith("0x")


def test_phase0_settings():
    assert len(BRIDGE_CONFIG["bridges"]) == 3
    assert CYCLE_FINDER_CONFIG["top_k"] == 32
    assert INVENTORY_DEFAULT["l1_eth"] > 0
    assert INVENTORY_DEFAULT["l2_eth"] > 0


def test_multipool_types_instantiate():
    from src.types import GasState

    pool = PoolSnapshot(
        chain="ethereum",
        pool_address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        token0="USDC",
        token1="WETH",
        fee_tier=500,
        sqrt_price_x96=1,
        liquidity=1,
        tick=0,
        eth_usdc_price=3500.0,
        pool_id="eth_usdc_005",
    )
    inv = InventoryState(l1_eth=10.0, l2_eth=8.0)
    assert inv.max_amount_in_eth == 8.0

    snap = MultiPoolSnapshot(
        snapshot_id="test_001",
        l1_block=1,
        l2_block=2,
        l1_pools=[pool],
        l2_pools=[pool],
        gas=GasState(l1_gwei=20.0, l2_gwei=0.1),
        bridge_fee_usd=5.0,
        inventory=inv,
        trigger="test",
        timestamp="2025-01-01T00:00:00Z",
    )
    assert snap.num_l1_pools == 1

    route = CandidateRoute(
        route_id=0,
        l1_pool_idx=0,
        l2_pool_idx=1,
        bridge_idx=2,
        score=1.5,
    )
    assert route.bridge_idx == 2


def test_load_registry_invalid_chain():
    with pytest.raises(ValueError, match="Unsupported chain"):
        load_registry("polygon")
