"""E2 多池多 GPU benchmark：对比 1/2/4 卡完整 PSO 墙钟。

本地 CPU 冒烟（单卡）:
    python scripts/benchmark_multipool_multi_gpu.py --gpus 1 --repeats 1 --particles 2000 --max-iter 40

云主机（需 Linux + 多 GPU + NCCL）:
    python scripts/benchmark_multipool_multi_gpu.py --gpus 1,2,4 --particles 8000 --max-iter 80 --repeats 3

说明: 老师版 E2 先做单卡 batch fitness 规模扫描；仅当 100k×K=32 墙钟 >2s 才「必须」跑多卡。
本脚本测的是**完整 PSO 迭代**的多卡扩展，可作为补充实验或云主机演示。
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
E2_DIR = ROOT / "data" / "E2-data"
RUN_DDP = ROOT / "scripts" / "run_ddp_multipool.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _run_cmd(nproc: int, record_id: str, top_k: int, particles: int, max_iter: int, seed: int, out_json: Path) -> list[str]:
    base = [
        str(RUN_DDP),
        "--record-id",
        record_id,
        "--top-k",
        str(top_k),
        "--num-particles",
        str(particles),
        "--max-iter",
        str(max_iter),
        "--seed",
        str(seed),
        "--json-output",
        str(out_json),
    ]
    if nproc == 1:
        return [sys.executable, *base]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        *base,
    ]


def _run_once(
    nproc: int,
    record_id: str,
    top_k: int,
    particles: int,
    max_iter: int,
    seed: int,
    run_idx: int,
) -> dict:
    out_json = E2_DIR / f"ddp_multipool_{nproc}gpu_r{run_idx}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = _run_cmd(nproc, record_id, top_k, particles, max_iter, seed, out_json)
    logger.info("运行: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def _plot(results: list[dict], out_path: Path) -> None:
    gpus = [r["world_size"] for r in results]
    times = [r["elapsed_ms_mean"] for r in results]
    base = times[0]
    speedups = [base / t for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar([str(g) for g in gpus], times, color=["#4C78A8", "#F58518", "#54A24B"][: len(gpus)])
    ax1.set_xlabel("Number of GPUs")
    ax1.set_ylabel("Wall-clock (ms)")
    ax1.set_title("Multipool PSO Wall-clock (E2 multi-GPU)")
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{t:.0f}", ha="center", va="bottom")

    ax2.plot(gpus, speedups, "o-", linewidth=2, label="Measured")
    ax2.plot(gpus, [g / gpus[0] for g in gpus], "--", alpha=0.7, label="Ideal linear")
    ax2.set_xlabel("Number of GPUs")
    ax2.set_ylabel("Speedup vs 1 GPU")
    ax2.set_title("Speedup")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="E2 multipool multi-GPU benchmark")
    parser.add_argument("--gpus", type=str, default="1,2,4")
    parser.add_argument("--record-id", type=str, default="mock_mp_001")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--particles", type=int, default=8000)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    summary_rows: list[dict] = []

    for nproc in gpu_list:
        elapsed_runs: list[float] = []
        fitness_runs: list[float] = []
        last_payload: dict | None = None

        for run_idx in range(args.repeats):
            payload = _run_once(
                nproc,
                args.record_id,
                args.top_k,
                args.particles,
                args.max_iter,
                args.seed,
                run_idx,
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
            "num_candidates": last_payload["num_candidates"] if last_payload else args.top_k,
            "max_iter": args.max_iter,
            "record_id": args.record_id,
        }
        if nproc == gpu_list[0]:
            row["speedup"] = 1.0
            row["parallel_efficiency_pct"] = 100.0
        else:
            base_time = summary_rows[0]["elapsed_ms_mean"]
            row["speedup"] = base_time / row["elapsed_ms_mean"]
            row["parallel_efficiency_pct"] = row["speedup"] / nproc * 100.0
        summary_rows.append(row)

        logger.info(
            "GPU=%d | mean=%.1f ms | speedup=%.2fx | eff=%.1f%% | fitness=%.2f",
            nproc,
            row["elapsed_ms_mean"],
            row.get("speedup", 1.0),
            row.get("parallel_efficiency_pct", 100.0),
            row["best_fitness_mean"],
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E2_multipool_multi_gpu",
        "record_id": args.record_id,
        "top_k": args.top_k,
        "particles": args.particles,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "repeats": args.repeats,
        "results": summary_rows,
        "notes": [
            "Full PSO search wall-clock (not single fitness batch).",
            "Teacher E2 gate: skip multi-GPU if single-card batch fitness 100k×K=32 < 2s.",
        ],
    }

    json_path = E2_DIR / "multipool_multi_gpu_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fig_path = FIG_DIR / "e2_multipool_multi_gpu.png"
    _plot(summary_rows, fig_path)

    logger.info("JSON: %s", json_path)
    logger.info("Figure: %s", fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
