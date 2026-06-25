"""
E3c：固定参数 vs 自适应 PSO 对比实验。
运行: python tests/test_adaptive_pso.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PSO_CONFIG
from src.data_loader import load_mock_snapshots
from src.models import CostModel, build_strategy, get_search_bounds
from src.optimizer import create_pso_optimizer
from src.pso_profile import get_fixed_pso_profile, select_pso_profile

FIG_DIR = ROOT / "data" / "figures"
RESULT_DIR = ROOT / "data" / "results"
SEED = 42


def _run_with_profile(snapshot, profile: dict):
    device = PSO_CONFIG["device"]
    bounds = get_search_bounds(device)
    cost_model = CostModel()
    optimizer = create_pso_optimizer(
        num_particles=profile["num_particles"],
        bounds=bounds,
        device=device,
        w=PSO_CONFIG["w"],
        c1=PSO_CONFIG["c1"],
        c2=PSO_CONFIG["c2"],
        seed=SEED,
    )

    def fitness_fn(positions, snap=snapshot):
        return cost_model.evaluate_fitness(positions, snap)

    result = optimizer.search(fitness_fn=fitness_fn, max_iter=profile["max_iter"])
    strategy = build_strategy(snapshot, result)
    return strategy, result, profile


def run_benchmark():
    snapshots = load_mock_snapshots()
    device = PSO_CONFIG["device"]
    print(f"设备: {device}")
    print(f"对比 {len(snapshots)} 条 Mock 快照\n")

    rows = []
    fixed_total_ms = 0.0
    adaptive_total_ms = 0.0

    for snapshot in snapshots:
        fixed_profile = get_fixed_pso_profile()
        fixed_profile["spread_pct"] = select_pso_profile(snapshot)["spread_pct"]
        fixed_profile["l2_lag_ms"] = snapshot.l2_lag_ms

        adaptive_profile = select_pso_profile(snapshot)

        fixed_strategy, fixed_result, _ = _run_with_profile(snapshot, fixed_profile)
        adaptive_strategy, adaptive_result, _ = _run_with_profile(snapshot, adaptive_profile)

        fixed_total_ms += fixed_result.elapsed_ms
        adaptive_total_ms += adaptive_result.elapsed_ms

        row = {
            "snapshot_id": snapshot.snapshot_id,
            "spread_pct": adaptive_profile["spread_pct"],
            "l2_lag_ms": snapshot.l2_lag_ms,
            "adaptive_profile": adaptive_profile["name"],
            "fixed": {
                "num_particles": fixed_profile["num_particles"],
                "max_iter": fixed_profile["max_iter"],
                "elapsed_ms": round(fixed_result.elapsed_ms, 2),
                "profit_usd": round(fixed_strategy.expected_profit_usd, 2),
            },
            "adaptive": {
                "num_particles": adaptive_profile["num_particles"],
                "max_iter": adaptive_profile["max_iter"],
                "elapsed_ms": round(adaptive_result.elapsed_ms, 2),
                "profit_usd": round(adaptive_strategy.expected_profit_usd, 2),
            },
        }
        rows.append(row)

        print(
            f"{snapshot.snapshot_id} | profile={adaptive_profile['name']:13s} | "
            f"fixed {fixed_result.elapsed_ms:7.1f}ms ${fixed_strategy.expected_profit_usd:8.2f} | "
            f"adaptive {adaptive_result.elapsed_ms:7.1f}ms ${adaptive_strategy.expected_profit_usd:8.2f}"
        )

    time_saved_pct = (
        (fixed_total_ms - adaptive_total_ms) / fixed_total_ms * 100 if fixed_total_ms else 0.0
    )
    summary = {
        "device": device,
        "snapshot_count": len(snapshots),
        "fixed_total_ms": round(fixed_total_ms, 2),
        "adaptive_total_ms": round(adaptive_total_ms, 2),
        "time_saved_pct": round(time_saved_pct, 2),
        "profile_counts": {},
        "records": rows,
    }
    for row in rows:
        name = row["adaptive_profile"]
        summary["profile_counts"][name] = summary["profile_counts"].get(name, 0) + 1

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / "adaptive_pso_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n总耗时 fixed={fixed_total_ms:.1f}ms adaptive={adaptive_total_ms:.1f}ms "
          f"(节省 {time_saved_pct:.1f}%)")
    print(f"JSON: {out_json}")

    _plot(rows, fixed_total_ms, adaptive_total_ms)
    return summary


def _plot(rows, fixed_total_ms, adaptive_total_ms):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ids = [r["snapshot_id"] for r in rows]
    fixed_times = [r["fixed"]["elapsed_ms"] for r in rows]
    adaptive_times = [r["adaptive"]["elapsed_ms"] for r in rows]
    fixed_profits = [r["fixed"]["profit_usd"] for r in rows]
    adaptive_profits = [r["adaptive"]["profit_usd"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = range(len(ids))
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], fixed_times, width, label="固定参数")
    axes[0].bar([i + width / 2 for i in x], adaptive_times, width, label="自适应")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(ids, rotation=45, ha="right")
    axes[0].set_ylabel("耗时 (ms)")
    axes[0].set_title("各快照 PSO 耗时")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar([0 - width / 2], [fixed_total_ms], width, label="固定参数")
    axes[1].bar([0 + width / 2], [adaptive_total_ms], width, label="自适应")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["合计"])
    axes[1].set_ylabel("总耗时 (ms)")
    axes[1].set_title("10 条 Mock 总墙钟时间")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = FIG_DIR / "adaptive_vs_fixed.png"
    plt.savefig(out_path, dpi=150)
    print(f"图表: {out_path}")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar([i - width / 2 for i in x], fixed_profits, width, label="固定参数")
    ax2.bar([i + width / 2 for i in x], adaptive_profits, width, label="自适应")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(ids, rotation=45, ha="right")
    ax2.set_ylabel("净利润 (USD)")
    ax2.set_title("各快照最优净利润")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    profit_path = FIG_DIR / "adaptive_vs_fixed_profit.png"
    plt.savefig(profit_path, dpi=150)
    print(f"图表: {profit_path}")


if __name__ == "__main__":
    run_benchmark()
