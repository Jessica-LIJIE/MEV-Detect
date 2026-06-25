"""E1：批量跑全部 Mock 多池记录并汇总 quality_ratio 分布。

示例:
    python scripts/run_e1_all_mock.py
    python scripts/run_e1_all_mock.py --particles 1000 --repeats 3 --warmup 3
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 all-mock solver comparison batch")
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    records = load_mock_multipool_records(MOCK_MULTIPOOL_DIR / "records.json")
    device = PSO_CONFIG["device"]
    rows: list[dict] = []

    print(f"\n=== E1 All Mock ({len(records)} records) ===\n")

    for item in records:
        record_id = item.snapshot.snapshot_id
        snapshot = item.snapshot
        routes = prepare_routes(snapshot, top_k=args.top_k)
        closed_profit, closed_amount, _ = closed_form_profit(
            snapshot,
            routes,
            latency_risk_lambda=item.latency_risk_lambda,
        )

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

        pso_times, _, pso_best = _run_with_warmup(pso_fn, args.warmup, args.repeats)
        ga_times, _, ga_best = _run_with_warmup(ga_fn, args.warmup, args.repeats)

        pso_mean = statistics.mean(pso_times)
        pso_std = statistics.pstdev(pso_times) if len(pso_times) > 1 else 0.0
        ga_mean = statistics.mean(ga_times)
        ga_std = statistics.pstdev(ga_times) if len(ga_times) > 1 else 0.0
        quality_ratio = pso_best / closed_profit if closed_profit > 0 else None
        speedup = ga_mean / pso_mean if pso_mean > 0 else None

        row = {
            "record_id": record_id,
            "category": item.category,
            "scenario": item.scenario,
            "closed_form_profit": closed_profit,
            "closed_form_amount_eth": closed_amount,
            "pso_best_fitness": pso_best,
            "ga_best_fitness": ga_best,
            "pso_ms_mean": pso_mean,
            "pso_ms_stdev": pso_std,
            "ga_ms_mean": ga_mean,
            "ga_ms_stdev": ga_std,
            "quality_ratio": quality_ratio,
            "ga_over_pso_speedup": speedup,
        }
        rows.append(row)

        qr = f"{quality_ratio:.4f}" if quality_ratio is not None else "—"
        print(
            f"{record_id} [{item.category}] PSO=${pso_best:,.2f} closed=${closed_profit:,.2f} "
            f"qr={qr} GA/PSO={speedup:.0f}x"
        )

    positive_qr = [r["quality_ratio"] for r in rows if r["quality_ratio"] is not None]
    qr_summary = {
        "count": len(positive_qr),
        "min": min(positive_qr) if positive_qr else None,
        "max": max(positive_qr) if positive_qr else None,
        "mean": statistics.mean(positive_qr) if positive_qr else None,
    }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E1_multipool_solver_compare_all",
        "particles": args.particles,
        "max_iter": args.max_iter,
        "device": device,
        "quality_ratio_summary": qr_summary,
        "records": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "e1_solver_compare_all.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = [r["record_id"].replace("mock_mp_", "") for r in rows]
    qr_vals = [r["quality_ratio"] if r["quality_ratio"] is not None else 0 for r in rows]
    colors = ["#54A24B" if r["quality_ratio"] and r["quality_ratio"] >= 0.95 else "#E45756" for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, qr_vals, color=colors)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="0.95 threshold")
    ax.set_xlabel("Mock record")
    ax.set_ylabel("quality_ratio (PSO / closed-form)")
    ax.set_title("E1 Solution quality across Mock multipool records")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig_path = FIG_DIR / "e1_quality_ratio_all.png"
    fig.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nquality_ratio: min={qr_summary['min']:.4f} max={qr_summary['max']:.4f} mean={qr_summary['mean']:.4f}")
    print(f"JSON: {out_json}")
    print(f"Figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
