"""E1：多池 PSO vs GA vs 闭式上界对照。

示例:
    python scripts/run_e1_solver_compare.py --record-id mock_mp_001
    python scripts/run_e1_solver_compare.py --record-id mock_mp_003 --particles 2000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

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

FIG_DIR = ROOT / "data" / "figures"
RESULTS_DIR = ROOT / "data" / "results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 multipool solver comparison")
    parser.add_argument("--record-id", type=str, default="mock_mp_001")
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3, help="预热次数（不计入统计）")
    parser.add_argument("--repeats", type=int, default=3, help="统计重复次数")
    parser.add_argument("--top-k", type=int, default=32)
    return parser.parse_args()


def _load_record(record_id: str):
    records = load_mock_multipool_records(MOCK_MULTIPOOL_DIR / "records.json")
    matched = [r for r in records if r.snapshot.snapshot_id == record_id]
    if not matched:
        raise ValueError(f"未找到 mock 记录: {record_id}")
    return matched[0]


def _run_with_warmup(fn, warmup: int, repeats: int) -> tuple[list[float], list[float], float]:
    times: list[float] = []
    profits: list[float] = []
    for _ in range(warmup):
        fn()
    for _ in range(repeats):
        result = fn()
        times.append(result.elapsed_ms)
        profits.append(result.best_fitness)
    return times, profits, profits[-1]


def main() -> int:
    args = _parse_args()
    item = _load_record(args.record_id)
    snapshot = item.snapshot
    routes = prepare_routes(snapshot, top_k=args.top_k)
    if not routes:
        print("No candidates for stage-1")
        return 1

    closed_profit, closed_amount, closed_route = closed_form_profit(
        snapshot,
        routes,
        latency_risk_lambda=item.latency_risk_lambda,
    )

    device = PSO_CONFIG["device"]
    print(f"\n=== E1 Solver Compare ===")
    print(f"record: {args.record_id} | category: {item.category}")
    print(f"scenario: {item.scenario}")
    print(f"candidates K={len(routes)} | device={device}")
    print(f"closed-form optimum: ${closed_profit:,.2f} @ {closed_amount:.4f} ETH")
    if closed_route:
        print(f"  route: L1[{closed_route.l1_pool_idx}] x L2[{closed_route.l2_pool_idx}] "
              f"bridge={closed_route.bridge_id}")
    print()

    def pso_fn():
        return run_pso_multipool(
            snapshot,
            routes,
            num_particles=args.particles,
            max_iter=args.max_iter,
            seed=args.seed,
            device=device,
        )

    def ga_fn():
        return run_ga_multipool(
            snapshot,
            routes,
            pop_size=args.particles,
            max_iter=args.max_iter,
            seed=args.seed,
            device="cpu",
        )

    pso_times, pso_profits, pso_best = _run_with_warmup(pso_fn, args.warmup, args.repeats)
    ga_times, ga_profits, ga_best = _run_with_warmup(ga_fn, args.warmup, args.repeats)

    pso_time_mean = statistics.mean(pso_times)
    pso_time_std = statistics.pstdev(pso_times) if len(pso_times) > 1 else 0.0
    ga_time_mean = statistics.mean(ga_times)
    ga_time_std = statistics.pstdev(ga_times) if len(ga_times) > 1 else 0.0

    quality_ratio = pso_best / closed_profit if closed_profit > 0 else None

    print(f"PSO: ${pso_best:,.2f} | {pso_time_mean:.1f}±{pso_time_std:.1f} ms")
    print(f"GA:  ${ga_best:,.2f} | {ga_time_mean:.1f}±{ga_time_std:.1f} ms")
    if quality_ratio is not None:
        print(f"quality_ratio (PSO/closed-form): {quality_ratio:.4f}")
    print(f"speedup GA/PSO wall-clock: {ga_time_mean / pso_time_mean:.2f}x")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    solvers = ["PSO", "GA", "Closed-form"]
    times = [pso_time_mean, ga_time_mean, 0.0]
    profits = [pso_best, ga_best, closed_profit]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    axes[0].bar(solvers, times, color=colors)
    axes[0].set_ylabel("Wall-clock (ms)")
    axes[0].set_title(f"E1 Timing ({args.record_id})")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(solvers, profits, color=colors)
    axes[1].set_ylabel("Net profit (USD)")
    axes[1].set_title("E1 Best profit")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = FIG_DIR / "e1_solver_compare.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E1_multipool_solver_compare",
        "record_id": args.record_id,
        "category": item.category,
        "scenario": item.scenario,
        "num_candidates": len(routes),
        "particles": args.particles,
        "max_iter": args.max_iter,
        "device": device,
        "closed_form_profit": closed_profit,
        "closed_form_amount_eth": closed_amount,
        "pso": {
            "best_fitness": pso_best,
            "elapsed_ms_mean": pso_time_mean,
            "elapsed_ms_stdev": pso_time_std,
            "runs_ms": pso_times,
        },
        "ga": {
            "best_fitness": ga_best,
            "elapsed_ms_mean": ga_time_mean,
            "elapsed_ms_stdev": ga_time_std,
            "runs_ms": ga_times,
        },
        "quality_ratio": quality_ratio,
        "figure": str(fig_path.relative_to(ROOT)),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "e1_solver_compare.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nJSON: {out_json}")
    print(f"Figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
