"""Phase 3：批量 [N,K] fitness、库存/时延、闭式上界。"""

from __future__ import annotations

import time

import pytest
import torch

from config.settings import BRIDGE_CONFIG, PSO_CONFIG
from src.closed_form import closed_form_upper_bound, optimize_route_amount
from src.cycle_finder import find_top_k_candidates
from src.models import MultiPoolCostModel, get_multipool_search_bounds
from src.optimizer import create_pso_optimizer
from src.swap_simulator import estimate_cross_layer_gross_usd
from src.types import CandidateRoute, GasState, InventoryState, MultiPoolSnapshot, PoolSnapshot


def _pool(
    chain: str,
    pool_id: str,
    *,
    price: float,
    liquidity: int = 3_000_000_000_000_000_000,
    token0: str = "USDC",
    token1: str = "WETH",
) -> PoolSnapshot:
    return PoolSnapshot(
        chain=chain,
        pool_address="0x" + "0" * 40,
        token0=token0,
        token1=token1,
        fee_tier=500,
        sqrt_price_x96=1,
        liquidity=liquidity,
        tick=0,
        eth_usdc_price=price,
        pool_id=pool_id,
    )


def _spread_snapshot() -> MultiPoolSnapshot:
    l1_pools = [
        _pool("ethereum", "eth_usdc_005", price=3550.0),
        _pool("ethereum", "eth_usdc_030", price=3540.0),
    ]
    l2_pools = [
        _pool("arbitrum", "arb_usdc_005", price=3500.0),
        _pool("arbitrum", "arb_usdc_030", price=3490.0),
    ]
    return MultiPoolSnapshot(
        snapshot_id="batch_test",
        l1_block=1,
        l2_block=2,
        l1_pools=l1_pools,
        l2_pools=l2_pools,
        gas=GasState(l1_gwei=20.0, l2_gwei=0.1),
        bridge_fee_usd=5.0,
        inventory=InventoryState(l1_eth=50.0, l2_eth=50.0),
        trigger="test",
        timestamp="2025-01-01T00:00:00Z",
        l2_lag_ms=500,
    )


def _route(i: int, j: int, b: int = 0) -> CandidateRoute:
    return CandidateRoute(
        route_id=0,
        l1_pool_idx=i,
        l2_pool_idx=j,
        bridge_idx=b,
        score=1.0,
        l1_pool_id=f"l1_{i}",
        l2_pool_id=f"l2_{j}",
        bridge_id=BRIDGE_CONFIG["bridges"][b]["id"],
    )


def test_profit_matrix_shape():
    snapshot = _spread_snapshot()
    routes = [_route(0, 0), _route(0, 1), _route(1, 0)]
    model = MultiPoolCostModel.from_routes(snapshot, routes)
    positions = torch.tensor([[10.0, 0.0], [20.0, 1.5], [5.0, 2.2]])
    matrix = model.evaluate_profit_matrix(positions, snapshot)
    assert matrix.shape == (3, 3)


def test_inventory_caps_amount():
    snapshot = _spread_snapshot()
    snapshot.inventory = InventoryState(l1_eth=5.0, l2_eth=5.0)
    route = _route(0, 0)
    model = MultiPoolCostModel.from_routes(snapshot, [route])
    profit_capped = model.evaluate_route_net_scalar(50.0, route, snapshot)
    profit_exact = model.evaluate_route_net_scalar(5.0, route, snapshot)
    assert profit_capped == pytest.approx(profit_exact)


def test_slow_bridge_lower_fitness_than_fast_bridge():
    snapshot = _spread_snapshot()
    fast = _route(0, 0, b=1)  # across
    slow = _route(0, 0, b=0)  # canonical
    model = MultiPoolCostModel.from_routes(snapshot, [fast, slow])
    amount = 10.0
    fast_profit = model.evaluate_route_net_scalar(amount, fast, snapshot)
    slow_profit = model.evaluate_route_net_scalar(amount, slow, snapshot)
    assert fast_profit > slow_profit


def test_evaluate_fitness_matches_matrix_column():
    snapshot = _spread_snapshot()
    routes = [_route(0, 0), _route(1, 1)]
    model = MultiPoolCostModel.from_routes(snapshot, routes)
    positions = torch.tensor([[15.0, 1.0], [8.0, 0.0]])
    fitness = model.evaluate_fitness(positions, snapshot)
    matrix = model.evaluate_profit_matrix(positions, snapshot)
    assert fitness[0].item() == pytest.approx(matrix[0, 1].item())
    assert fitness[1].item() == pytest.approx(matrix[1, 0].item())


def test_closed_form_beats_small_amount():
    snapshot = _spread_snapshot()
    route = _route(0, 0, b=1)
    model = MultiPoolCostModel.from_routes(snapshot, [route])
    opt_amount, opt_profit = optimize_route_amount(route, snapshot, model)
    small_profit = model.evaluate_route_net_scalar(0.5, route, snapshot)
    assert opt_profit >= small_profit
    assert opt_amount > 0.1


def test_closed_form_upper_bound_dict():
    snapshot = _spread_snapshot()
    route = _route(0, 0)
    model = MultiPoolCostModel.from_routes(snapshot, [route])
    result = closed_form_upper_bound(route, snapshot, model)
    assert "amount_in_eth" in result
    assert "net_profit_usd" in result


def test_swap_simulator_gross_positive_on_spread():
    snapshot = _spread_snapshot()
    gross = estimate_cross_layer_gross_usd(
        10.0, snapshot.l1_pools[0], snapshot.l2_pools[0]
    )
    assert gross > 0


def test_multipool_pso_runs():
    snapshot = _spread_snapshot()
    routes = find_top_k_candidates(snapshot, top_k=4, min_effective_depth_usd=1.0)
    model = MultiPoolCostModel.from_routes(snapshot, routes)
    bounds = get_multipool_search_bounds(snapshot, len(routes), "cpu")
    optimizer = create_pso_optimizer(
        num_particles=64,
        bounds=bounds,
        device="cpu",
        seed=1,
    )
    result = optimizer.search(
        fitness_fn=lambda pos: model.evaluate_fitness(pos, snapshot),
        max_iter=10,
    )
    assert result.best_fitness is not None
    assert len(result.best_position) == 2


@pytest.mark.slow
def test_batch_fitness_large_scale_cpu():
    """100k 粒子 × K=32：验证可向量化跑通（CPU 墙钟参考）。"""
    snapshot = _spread_snapshot()
    routes = [_route(i % 2, i % 2, i % 3) for i in range(32)]
    for idx, r in enumerate(routes):
        r.route_id = idx
    model = MultiPoolCostModel.from_routes(snapshot, routes)
    n, k = 100_000, 32
    bounds = get_multipool_search_bounds(snapshot, k, "cpu")
    low, high = bounds[:, 0], bounds[:, 1]
    positions = torch.rand(n, 2) * (high - low) + low

    start = time.perf_counter()
    matrix = model.evaluate_profit_matrix(positions, snapshot)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert matrix.shape == (n, k)
    assert elapsed_ms > 0
    print(f"batch fitness {n}x{k}: {elapsed_ms:.1f} ms on cpu")
