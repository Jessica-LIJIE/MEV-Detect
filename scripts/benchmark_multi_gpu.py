"""E2 多 GPU benchmark：对比 1/2/4 卡墙钟耗时并出图。

云主机示例:
    python scripts/benchmark_multi_gpu.py --gpus 1,2,4 --record-id snap_003

本地单进程冒烟:
    python scripts/benchmark_multi_gpu.py --gpus 1 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "data" / "figures"
RESULTS_DIR = ROOT / "data" / "results"
RUN_DDP = ROOT / "scripts" / "run_ddp.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _torchrun_cmd(nproc: int, record_id: str, particles: int, max_iter: int, seed: int, out_json: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        str(RUN_DDP),
        "--mock",
        "--record-id",
        record_id,
        "--num-particles",
        str(particles),
        "--max-iter",
        str(max_iter),
        "--seed",
        str(seed),
        "--json-output",
        str(out_json),
    ]


def _run_once(
    nproc: int,
    record_id: str,
    particles: int,
    max_iter: int,
    seed: int,
    run_idx: int,
) -> dict:
    out_json = RESULTS_DIR / f"ddp_run_{nproc}gpu_r{run_idx}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if nproc == 1:
        cmd = [
            sys.executable,
            str(RUN_DDP),
            "--mock",
            "--record-id",
            record_id,
            "--num-particles",
            str(particles),
            "--max-iter",
            str(max_iter),
            "--seed",
            str(seed),
            "--json-output",
            str(out_json),
        ]
    else:
        cmd = _torchrun_cmd(nproc, record_id, particles, max_iter, seed, out_json)

    logger.info("运行: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def _plot(results: list[dict], out_speedup: Path, out_efficiency: Path) -> None:
    gpus = [r["world_size"] for r in results]
    times = [r["elapsed_ms_mean"] for r in results]
    base = times[0]
    speedups = [base / t for t in times]
    efficiencies = [s / g * 100 for s, g in zip(speedups, gpus)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([str(g) for g in gpus], times, color=["#4C78A8", "#F58518", "#54A24B"][: len(gpus)])
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("Multi-GPU PSO Wall-clock Time (E2)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{t:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    out_speedup.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_speedup, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gpus, speedups, "o-", label="Speedup", linewidth=2)
    ax.plot(gpus, [g / gpus[0] for g in gpus], "--", label="Ideal linear", alpha=0.7)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup")
    ax.set_title("Multi-GPU Speedup & Parallel Efficiency")
    ax2_twin = ax.twinx()
    ax2_twin.bar([g - 0.15 for g in gpus], efficiencies, width=0.3, alpha=0.35, color="#72B7B2", label="Efficiency %")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_efficiency, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 多 GPU benchmark")
    parser.add_argument("--gpus", type=str, default="1,2,4", help="逗号分隔 GPU 数，如 1,2,4")
    parser.add_argument("--record-id", type=str, default="snap_003")
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3, help="每种 GPU 配置重复次数")
    args = parser.parse_args()

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    summary_rows: list[dict] = []

    for nproc in gpu_list:
        elapsed_runs: list[float] = []
        fitness_runs: list[float] = []
        last_payload: dict | None = None

        for run_idx in range(args.repeats):
            payload = _run_once(
                nproc=nproc,
                record_id=args.record_id,
                particles=args.particles,
                max_iter=args.max_iter,
                seed=args.seed,
                run_idx=run_idx,
            )
            elapsed_runs.append(float(payload["elapsed_ms"]))
            fitness_runs.append(float(payload["best_fitness"]))
            last_payload = payload

        row = {
            "world_size": nproc,
            "elapsed_ms_mean": statistics.mean(elapsed_runs),
            "elapsed_ms_stdev": statistics.pstdev(elapsed_runs) if len(elapsed_runs) > 1 else 0.0,
            "elapsed_ms_runs": elapsed_runs,
            "best_fitness_mean": statistics.mean(fitness_runs),
            "num_particles_total": args.particles,
            "num_particles_per_rank": last_payload["num_particles_per_rank"] if last_payload else None,
            "record_id": args.record_id,
            "max_iter": args.max_iter,
            "seed": args.seed,
        }
        if nproc == gpu_list[0]:
            row["speedup"] = 1.0
            row["parallel_efficiency_pct"] = 100.0
        else:
            base_time = summary_rows[0]["elapsed_ms_mean"]
            row["speedup"] = base_time / row["elapsed_ms_mean"]
            row["parallel_efficiency_pct"] = row["speedup"] / nproc * 100
        summary_rows.append(row)

        logger.info(
            "GPU=%d | mean=%.2f ms | speedup=%.2fx | fitness=%.2f",
            nproc,
            row["elapsed_ms_mean"],
            row.get("speedup", 1.0),
            row["best_fitness_mean"],
        )

    benchmark = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_id": args.record_id,
        "particles": args.particles,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "repeats": args.repeats,
        "results": summary_rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_path = RESULTS_DIR / "multi_gpu_benchmark.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    logger.info("Benchmark 数据已写入 %s", bench_path)

    speedup_png = FIG_DIR / "multi_gpu_speedup.png"
    efficiency_png = FIG_DIR / "multi_gpu_efficiency.png"
    _plot(summary_rows, speedup_png, efficiency_png)
    logger.info("图表已保存: %s , %s", speedup_png, efficiency_png)


if __name__ == "__main__":
    main()
