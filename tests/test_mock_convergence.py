"""Phase 4：Mock 多池消融 + PSO 与闭式最优收敛。"""

from __future__ import annotations

import pytest

from config.settings import PSO_CONFIG
from src.closed_form import global_closed_form_optimum
from src.cycle_finder import find_top_k_candidates
from src.models import MultiPoolCostModel, get_multipool_search_bounds
from src.multipool_mock import load_mock_multipool_records
from src.optimizer import create_pso_optimizer


def _records_by_category(category: str):
    return [r for r in load_mock_multipool_records() if r.category == category]


def test_mock_multipool_record_count():
    records = load_mock_multipool_records()
    assert 7 <= len(records) <= 10
    categories = {r.category for r in records}
    assert "profitable_pair" in categories
    assert "no_opportunity" in categories
    assert "latency_eats_profit" in categories


def test_profitable_records_have_positive_closed_form():
    for item in _records_by_category("profitable_pair"):
        snapshot = item.snapshot
        routes = find_top_k_candidates(snapshot, top_k=32, min_effective_depth_usd=1.0)
        best_profit, _, best_route = global_closed_form_optimum(routes, snapshot)
        assert best_profit > 0, f"{item.snapshot.snapshot_id} should be profitable"
        if item.best_pair is not None:
            top_pairs = {(r.l1_pool_idx, r.l2_pool_idx) for r in routes[:5]}
            assert item.best_pair in top_pairs, f"{item.snapshot.snapshot_id} missing best pair"


def test_no_opportunity_records_non_positive():
    for item in _records_by_category("no_opportunity"):
        snapshot = item.snapshot
        routes = find_top_k_candidates(snapshot, top_k=32, min_effective_depth_usd=1.0)
        best_profit, _, _ = global_closed_form_optimum(routes, snapshot)
        assert best_profit <= 0, f"{item.snapshot.snapshot_id} should have no profit"


def test_latency_records_elevated_lambda_hurts_canonical():
    item = next(r for r in load_mock_multipool_records() if r.snapshot.snapshot_id == "mock_mp_006")
    snapshot = item.snapshot
    routes = find_top_k_candidates(snapshot, top_k=32, min_effective_depth_usd=1.0)
    lam = item.latency_risk_lambda or 0.08
    slow_profit, _, slow_route = global_closed_form_optimum(
        [r for r in routes if r.bridge_idx == 0],
        snapshot,
        latency_risk_lambda=lam,
    )
    fast_profit, _, _ = global_closed_form_optimum(
        [r for r in routes if r.bridge_idx == 1],
        snapshot,
        latency_risk_lambda=lam,
    )
    assert slow_profit <= 0
    assert fast_profit > slow_profit


@pytest.mark.parametrize("record_id", ["mock_mp_001", "mock_mp_002", "mock_mp_003"])
def test_pso_within_five_percent_of_closed_form(record_id: str):
    """PSO 最优 fitness 与闭式全局最优误差 < 5%（盈利 Mock）。"""
    item = next(r for r in load_mock_multipool_records() if r.snapshot.snapshot_id == record_id)
    snapshot = item.snapshot
    routes = find_top_k_candidates(snapshot, top_k=32, min_effective_depth_usd=1.0)
    closed_profit, _, _ = global_closed_form_optimum(routes, snapshot)
    assert closed_profit > 0

    model = MultiPoolCostModel.from_routes(snapshot, routes)
    bounds = get_multipool_search_bounds(snapshot, len(routes), "cpu")
    optimizer = create_pso_optimizer(
        num_particles=2000,
        bounds=bounds,
        device="cpu",
        w=PSO_CONFIG["w"],
        c1=PSO_CONFIG["c1"],
        c2=PSO_CONFIG["c2"],
        seed=42,
    )
    result = optimizer.search(
        fitness_fn=lambda pos: model.evaluate_fitness(pos, snapshot),
        max_iter=120,
        patience=25,
    )

    rel_err = abs(result.best_fitness - closed_profit) / closed_profit
    assert rel_err < 0.05, (
        f"{record_id}: PSO={result.best_fitness:.2f} closed={closed_profit:.2f} err={rel_err:.2%}"
    )
