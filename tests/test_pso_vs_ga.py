"""
PSO vs GA 对比实验。
运行: python tests/test_pso_vs_ga.py
输出图表到 data/figures/
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PSO_CONFIG
from src.data_loader import load_mock_snapshots
from src.models import CostModel, get_search_bounds
from src.optimizer import GAOptimizer, PSOOptimizer

FIG_DIR = ROOT / "data" / "figures"
PARTICLE_COUNTS = [100, 300, 500, 1000]
BENCHMARK_SNAPSHOT_ID = "snap_004"
SEED = 42


def _run_once(optimizer_cls, num_particles, snapshot, device, max_iter=80):
    bounds = get_search_bounds(device)
    model = CostModel()

    def fitness_fn(positions):
        return model.evaluate_fitness(positions, snapshot)

    if optimizer_cls is PSOOptimizer:
        opt = PSOOptimizer(
            num_particles=num_particles,
            dim=4,
            bounds=bounds,
            device=device,
            seed=SEED,
        )
    else:
        opt = GAOptimizer(
            pop_size=num_particles,
            dim=4,
            bounds=bounds,
            device=device,
            seed=SEED,
        )

    return opt.search(fitness_fn=fitness_fn, max_iter=max_iter)


def run_benchmark():
    device = PSO_CONFIG["device"]
    snapshot = next(s for s in load_mock_snapshots() if s.snapshot_id == BENCHMARK_SNAPSHOT_ID)

    print(f"对比快照: {snapshot.snapshot_id} ({snapshot.scenario})")
    print(f"设备: {device}\n")

    pso_times, ga_times = [], []
    pso_profits, ga_profits = [], []

    for n in PARTICLE_COUNTS:
        print(f"粒子/种群数 = {n} ...", end=" ", flush=True)

        pso_result = _run_once(PSOOptimizer, n, snapshot, device)
        ga_result = _run_once(GAOptimizer, n, snapshot, device)

        pso_times.append(pso_result.elapsed_ms)
        ga_times.append(ga_result.elapsed_ms)
        pso_profits.append(pso_result.best_fitness)
        ga_profits.append(ga_result.best_fitness)

        print(
            f"PSO {pso_result.elapsed_ms:.1f}ms (${pso_result.best_fitness:.2f}) | "
            f"GA {ga_result.elapsed_ms:.1f}ms (${ga_result.best_fitness:.2f})"
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(PARTICLE_COUNTS, pso_times, "o-", label="PSO (向量化)", linewidth=2)
    axes[0].plot(PARTICLE_COUNTS, ga_times, "s-", label="GA (串行)", linewidth=2)
    axes[0].set_xlabel("粒子 / 种群数量")
    axes[0].set_ylabel("耗时 (ms)")
    axes[0].set_title("收敛耗时对比")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x = range(len(PARTICLE_COUNTS))
    width = 0.35
    axes[1].bar([i - width / 2 for i in x], pso_profits, width, label="PSO")
    axes[1].bar([i + width / 2 for i in x], ga_profits, width, label="GA")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([str(n) for n in PARTICLE_COUNTS])
    axes[1].set_xlabel("粒子 / 种群数量")
    axes[1].set_ylabel("最优净利润 (USD)")
    axes[1].set_title("最优利润对比")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = FIG_DIR / "pso_vs_ga.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n图表已保存: {out_path}")

    speedup = [ga / pso if pso > 0 else 0 for pso, ga in zip(pso_times, ga_times)]
    print("\n加速比 (GA耗时 / PSO耗时):")
    for n, s in zip(PARTICLE_COUNTS, speedup):
        print(f"  n={n}: {s:.2f}x")


if __name__ == "__main__":
    run_benchmark()
