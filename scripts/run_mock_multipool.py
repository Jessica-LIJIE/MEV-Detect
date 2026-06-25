"""CLI：批量运行 Mock 多池消融记录（Phase 4，仅 E1/E3）。

示例:
    python scripts/run_mock_multipool.py
    python scripts/run_mock_multipool.py --id mock_mp_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PSO_CONFIG
from src.closed_form import global_closed_form_optimum
from src.cycle_finder import find_top_k_candidates
from src.models import MultiPoolCostModel, build_multipool_strategy, get_multipool_search_bounds
from src.multipool_mock import load_mock_multipool_records
from src.optimizer import create_pso_optimizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mock multipool ablation records")
    parser.add_argument("--id", type=str, default=None, help="Run single record id")
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--max-iter", type=int, default=120)
    return parser.parse_args()


def run_record(record_id: str | None, *, particles: int, max_iter: int) -> int:
    records = load_mock_multipool_records()
    if record_id:
        records = [r for r in records if r.snapshot.snapshot_id == record_id]
        if not records:
            print(f"Record not found: {record_id}")
            return 1

    print(f"\n=== Mock Multipool Ablation ({len(records)} records) ===\n")
    for item in records:
        snapshot = item.snapshot
        routes = find_top_k_candidates(snapshot, top_k=32, min_effective_depth_usd=1.0)
        lam = item.latency_risk_lambda
        closed_profit, closed_amount, closed_route = global_closed_form_optimum(
            routes,
            snapshot,
            latency_risk_lambda=lam,
        )

        model = MultiPoolCostModel.from_routes(
            snapshot, routes, latency_risk_lambda=lam
        )
        bounds = get_multipool_search_bounds(snapshot, len(routes), "cpu")
        optimizer = create_pso_optimizer(
            num_particles=particles,
            bounds=bounds,
            device="cpu",
            w=PSO_CONFIG["w"],
            c1=PSO_CONFIG["c1"],
            c2=PSO_CONFIG["c2"],
            seed=42,
        )
        result = optimizer.search(
            fitness_fn=lambda pos, m=model, s=snapshot: m.evaluate_fitness(pos, s),
            max_iter=max_iter,
            patience=25,
        )
        strategy = build_multipool_strategy(snapshot, routes, result)
        rel_err = (
            abs(result.best_fitness - closed_profit) / closed_profit
            if closed_profit > 0
            else None
        )

        print(f"[{item.category}] {snapshot.snapshot_id}")
        print(f"  scenario: {item.scenario}")
        print(f"  closed-form: ${closed_profit:,.2f} @ {closed_amount:.4f} ETH", end="")
        if closed_route:
            print(
                f" | L1[{closed_route.l1_pool_idx}] x L2[{closed_route.l2_pool_idx}] "
                f"bridge={closed_route.bridge_id}"
            )
        else:
            print()
        print(f"  PSO: ${result.best_fitness:,.2f} | {strategy.summary()}")
        if rel_err is not None:
            print(f"  rel_err vs closed-form: {rel_err:.2%}")
        print()

    return 0


def main() -> int:
    args = _parse_args()
    return run_record(args.id, particles=args.particles, max_iter=args.max_iter)


if __name__ == "__main__":
    raise SystemExit(main())
