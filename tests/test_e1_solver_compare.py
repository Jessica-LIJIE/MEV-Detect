"""E1 多池 PSO vs GA vs 闭式上界（pytest 快速验收）。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import MOCK_MULTIPOOL_DIR, PSO_CONFIG
from src.experiments.multipool_runner import (
    closed_form_profit,
    prepare_routes,
    run_ga_multipool,
    run_pso_multipool,
)
from src.multipool_mock import load_mock_multipool_records


def _load_mock(record_id: str):
    records = load_mock_multipool_records(MOCK_MULTIPOOL_DIR / "records.json")
    matched = [r for r in records if r.snapshot.snapshot_id == record_id]
    if not matched:
        pytest.skip(f"record not found: {record_id}")
    return matched[0]


@pytest.mark.parametrize("record_id", ["mock_mp_001", "mock_mp_003"])
def test_e1_pso_near_closed_form(record_id: str):
    item = _load_mock(record_id)
    snapshot = item.snapshot
    routes = prepare_routes(snapshot, top_k=32)
    assert routes

    closed_profit, _, _ = closed_form_profit(
        snapshot,
        routes,
        latency_risk_lambda=item.latency_risk_lambda,
    )
    assert closed_profit > 0

    pso = run_pso_multipool(
        snapshot,
        routes,
        num_particles=500,
        max_iter=60,
        seed=42,
        device=PSO_CONFIG["device"],
    )
    quality_ratio = pso.best_fitness / closed_profit
    assert quality_ratio >= 0.95, f"PSO too far from closed-form: {quality_ratio:.3f}"


def test_e1_ga_runs_on_mock():
    item = _load_mock("mock_mp_001")
    routes = prepare_routes(item.snapshot, top_k=16)
    ga = run_ga_multipool(
        item.snapshot,
        routes,
        pop_size=200,
        max_iter=40,
        seed=42,
        device="cpu",
    )
    assert ga.elapsed_ms > 0
