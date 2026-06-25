"""阶段1 完整实现：token 图 + 负对数路径检测。"""

from __future__ import annotations

import pytest

from config.settings import CYCLE_FINDER_CONFIG
from src.cycle_finder import find_top_k_candidates, find_top_k_candidates_mvp
from src.stage1_graph import (
    build_token_graph,
    enumerate_cross_layer_paths,
    find_graph_candidates,
    negative_log_rate,
    pool_swap_rate,
)
from src.types import PoolSnapshot
from tests.test_cycle_finder import _mock_snapshot


def test_negative_log_rate_monotonic():
    assert negative_log_rate(1.0) == pytest.approx(0.0)
    assert negative_log_rate(2.0) < negative_log_rate(1.5)


def test_pool_swap_rate_respects_fee():
    pool = PoolSnapshot(
        chain="ethereum",
        pool_address="0x" + "0" * 40,
        token0="USDC",
        token1="WETH",
        fee_tier=500,
        sqrt_price_x96=1,
        liquidity=1,
        tick=0,
        eth_usdc_price=3500.0,
        pool_id="p",
    )
    weth_to_usdc = pool_swap_rate(pool, "WETH", "USDC")
    usdc_to_weth = pool_swap_rate(pool, "USDC", "WETH")
    assert weth_to_usdc == pytest.approx(3500.0 * 0.9995)
    assert usdc_to_weth == pytest.approx((1.0 / 3500.0) * 0.9995)


def test_graph_finds_cross_layer_paths():
    snapshot = _mock_snapshot([3500.0] * 6, [3500.0] * 6)
    adj = build_token_graph(snapshot)
    paths = enumerate_cross_layer_paths(adj, max_hops=6)
    assert len(paths) > 0
    assert all(p.l1_pool_idx >= 0 and p.l2_pool_idx >= 0 for p in paths)


def test_full_mode_contains_best_spread_pair():
    base = 3500.0
    l1_prices = [base] * 6
    l2_prices = [base] * 6
    l1_prices[2] = 3650.0
    l2_prices[5] = 3350.0
    snapshot = _mock_snapshot(l1_prices, l2_prices)

    routes = find_graph_candidates(snapshot, top_k=32)
    assert len(routes) > 0
    best_pairs = {(r.l1_pool_idx, r.l2_pool_idx) for r in routes[:3]}
    assert (2, 5) in best_pairs
    assert routes[0].l1_pool_idx == 2
    assert routes[0].l2_pool_idx == 5


def test_full_mode_matches_mvp_top_pair_on_spread():
    base = 3500.0
    l1_prices = [base] * 6
    l2_prices = [base] * 6
    l1_prices[2] = 3650.0
    l2_prices[5] = 3350.0
    snapshot = _mock_snapshot(l1_prices, l2_prices)

    full = find_top_k_candidates(snapshot, mode="full", top_k=32)
    mvp = find_top_k_candidates(snapshot, mode="mvp", top_k=32)
    assert full[0].l1_pool_idx == mvp[0].l1_pool_idx
    assert full[0].l2_pool_idx == mvp[0].l2_pool_idx


def test_admissible_relaxation_keeps_near_miss_route():
    """价差略小但仍接近盈利阈值的路由，松弛后应保留。"""
    base = 3500.0
    l1_prices = [base] * 6
    l2_prices = [base] * 6
    l1_prices[1] = 3500.8
    l2_prices[1] = 3499.5
    snapshot = _mock_snapshot(l1_prices, l2_prices)

    strict = find_graph_candidates(
        snapshot,
        top_k=64,
        admissible_relaxation_eps=0.0,
        depth_relaxation=0.0,
    )
    relaxed = find_graph_candidates(
        snapshot,
        top_k=64,
        admissible_relaxation_eps=0.01,
        depth_relaxation=0.2,
    )
    assert len(relaxed) >= len(strict)


def test_default_find_top_k_uses_full_mode():
    assert CYCLE_FINDER_CONFIG.get("mode") == "full"
    snapshot = _mock_snapshot([3500.0] * 6, [3500.0] * 6)
    routes = find_top_k_candidates(snapshot, top_k=8)
    mvp_routes = find_top_k_candidates_mvp(snapshot, top_k=8)
    assert len(routes) == len(mvp_routes)
