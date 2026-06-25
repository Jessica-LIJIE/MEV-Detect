"""Phase 2：阶段1 候选环 Top-K 筛选。"""

from __future__ import annotations

import pytest

from config.settings import CYCLE_FINDER_CONFIG
from src.cycle_finder import (
    effective_depth_usd,
    find_top_k_candidates,
    pair_spread_abs,
    static_route_score,
)
from src.types import GasState, InventoryState, MultiPoolSnapshot, PoolSnapshot


def _pool(
    chain: str,
    pool_id: str,
    *,
    price: float,
    liquidity: int = 3_000_000_000_000_000_000,
) -> PoolSnapshot:
    return PoolSnapshot(
        chain=chain,
        pool_address=f"0x{'0' * 40}",
        token0="USDC",
        token1="WETH",
        fee_tier=500,
        sqrt_price_x96=1,
        liquidity=liquidity,
        tick=0,
        eth_usdc_price=price,
        pool_id=pool_id,
    )


def _mock_snapshot(
    l1_prices: list[float],
    l2_prices: list[float],
    *,
    l1_liquidities: list[int] | None = None,
    l2_liquidities: list[int] | None = None,
) -> MultiPoolSnapshot:
    n1, n2 = len(l1_prices), len(l2_prices)
    l1_liq = l1_liquidities or [3_000_000_000_000_000_000] * n1
    l2_liq = l2_liquidities or [3_000_000_000_000_000_000] * n2
    return MultiPoolSnapshot(
        snapshot_id="mock_cycle",
        l1_block=1,
        l2_block=2,
        l1_pools=[
            _pool("ethereum", f"eth_pool_{i}", price=p, liquidity=l1_liq[i])
            for i, p in enumerate(l1_prices)
        ],
        l2_pools=[
            _pool("arbitrum", f"arb_pool_{j}", price=p, liquidity=l2_liq[j])
            for j, p in enumerate(l2_prices)
        ],
        gas=GasState(l1_gwei=20.0, l2_gwei=0.1),
        bridge_fee_usd=5.0,
        inventory=InventoryState(l1_eth=50.0, l2_eth=50.0),
        trigger="test",
        timestamp="2025-01-01T00:00:00Z",
    )


def test_effective_depth_scales_with_liquidity():
    deep = _pool("ethereum", "deep", price=3500.0, liquidity=3_000_000_000_000_000_000)
    shallow = _pool("ethereum", "shallow", price=3500.0, liquidity=100_000_000_000_000)
    assert effective_depth_usd(deep) > effective_depth_usd(shallow)


def test_depth_filter_excludes_shallow_pool():
    l1_prices = [3500.0] * 6
    l2_prices = [3500.0] * 6
    l1_liq = [3_000_000_000_000_000_000] * 6
    l2_liq = [3_000_000_000_000_000_000] * 6
    l1_liq[0] = 100_000_000_000_000  # 浅池，应被过滤

    snapshot = _mock_snapshot(l1_prices, l2_prices, l1_liquidities=l1_liq, l2_liquidities=l2_liq)
    routes = find_top_k_candidates(snapshot, top_k=64, min_effective_depth_usd=50_000.0)
    assert all(r.l1_pool_idx != 0 for r in routes)


def test_top_k_contains_best_spread_pair():
    """Mock 快照中仅 (i=2, j=5) 价差足够大 → Top-K 必含该对。"""
    base = 3500.0
    l1_prices = [base] * 6
    l2_prices = [base] * 6
    l1_prices[2] = 3650.0
    l2_prices[5] = 3350.0

    snapshot = _mock_snapshot(l1_prices, l2_prices)
    routes = find_top_k_candidates(snapshot, top_k=32)

    assert len(routes) <= 32
    assert len(routes) > 0
    best_pairs = {(r.l1_pool_idx, r.l2_pool_idx) for r in routes[:3]}
    assert (2, 5) in best_pairs
    assert routes[0].l1_pool_idx == 2
    assert routes[0].l2_pool_idx == 5
    assert routes[0].score >= routes[-1].score


def test_pair_spread_and_score_ordering():
    l1 = _pool("ethereum", "a", price=3600.0)
    l2 = _pool("arbitrum", "b", price=3400.0)
    l2_flat = _pool("arbitrum", "c", price=3590.0)

    assert pair_spread_abs(l1, l2) == pytest.approx(200.0)
    assert static_route_score(l1, l2) > static_route_score(l1, l2_flat)


def test_excludes_non_stable_cross_pairs():
    """WETH/ARB 等非稳定币池不参与跨层 stable 配对。"""
    l1_prices = [3500.0] * 6
    l2_prices = [3500.0] * 6
    l2_prices[5] = 5000.0  # 若误配对 ARB 池会产生虚假大价差

    snapshot = _mock_snapshot(l1_prices, l2_prices)
    snapshot.l2_pools[5] = PoolSnapshot(
        chain="arbitrum",
        pool_address="0x" + "1" * 40,
        token0="WETH",
        token1="ARB",
        fee_tier=500,
        sqrt_price_x96=1,
        liquidity=3_000_000_000_000_000_000,
        tick=0,
        eth_usdc_price=5000.0,
        pool_id="arb_arb_005",
    )

    routes = find_top_k_candidates(snapshot, top_k=32)
    assert all(r.l2_pool_idx != 5 for r in routes)


def test_respects_top_k_config():
    snapshot = _mock_snapshot([3500.0] * 6, [3500.0] * 6)
    routes = find_top_k_candidates(snapshot)
    assert len(routes) <= CYCLE_FINDER_CONFIG["top_k"]
