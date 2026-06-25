"""E2：多池单卡规模曲线 particles × K（CPU/CUDA）。

示例:
    python scripts/benchmark_multipool_scale.py
    python scripts/benchmark_multipool_scale.py --record-id mock_mp_001 --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import MOCK_MULTIPOOL_DIR, PSO_CONFIG
from src.experiments.multipool_runner import prepare_routes, timed_fitness_batch
from src.multipool_mock import load_mock_multipool_records

FIG_DIR = ROOT / "data" / "figures"
E2_DIR = ROOT / "data" / "E2-data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARTICLE_GRID_FULL = [10_000, 50_000, 100_000, 200_000]
PARTICLE_GRID_QUICK = [1_000, 5_000, 10_000, 20_000]
K_GRID = [16, 32, 64]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2 multipool scale benchmark")
    parser.add_argument("--record-id", type=str, default="mock_mp_001")
    parser.add_argument("--quick", action="store_true", help="Smaller particle grid for local CPU")
    parser.add_argument("--cpu-only", action="store_true", help="Skip CUDA even if available")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def _load_snapshot(record_id: str):
    records = load_mock_multipool_records(MOCK_MULTIPOOL_DIR / "records.json")
    for item in records:
        if item.snapshot.snapshot_id == record_id:
            return item.snapshot
    raise ValueError(f"record not found: {record_id}")


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.rand(4, device="cuda")
        probe = probe * 2
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def _devices(cpu_only: bool = False) -> list[str]:
    if cpu_only:
        return ["cpu"]
    devices = ["cpu"]
    if _cuda_usable():
        devices.append("cuda")
    return devices


def _plot_particles(rows: list[dict], out_path: Path, fixed_k: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for device in sorted({r["device"] for r in rows}):
        subset = [r for r in rows if r["device"] == device and r["k"] == fixed_k]
        if not subset:
            continue
        xs = [r["particles"] for r in subset]
        ys = [r["elapsed_ms_mean"] for r in subset]
        ax.plot(xs, ys, "o-", label=device, linewidth=2)
    ax.set_xlabel("Particles (N)")
    ax.set_ylabel("Fitness batch wall-clock (ms)")
    ax.set_title(f"E2 Scale vs Particles (K={fixed_k})")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_k(rows: list[dict], out_path: Path, fixed_n: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for device in sorted({r["device"] for r in rows}):
        subset = [r for r in rows if r["device"] == device and r["particles"] == fixed_n]
        if not subset:
            continue
        xs = [r["k"] for r in subset]
        ys = [r["elapsed_ms_mean"] for r in subset]
        ax.plot(xs, ys, "o-", label=device, linewidth=2)
    ax.set_xlabel("Candidates K")
    ax.set_ylabel("Fitness batch wall-clock (ms)")
    ax.set_title(f"E2 Scale vs K (N={fixed_n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    snapshot = _load_snapshot(args.record_id)
    particle_grid = PARTICLE_GRID_QUICK if args.quick else PARTICLE_GRID_FULL
    devices = _devices(args.cpu_only)

    rows: list[dict] = []
    logger.info("E2 multipool scale | record=%s | devices=%s", args.record_id, devices)

    for device in devices:
        for k in K_GRID:
            routes = prepare_routes(snapshot, top_k=k)
            if len(routes) < k:
                logger.warning("Only %d routes available for K=%d", len(routes), k)
            for n in particle_grid:
                mean_ms, std_ms = timed_fitness_batch(
                    snapshot,
                    routes,
                    num_particles=n,
                    device=device,
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                row = {
                    "device": device,
                    "particles": n,
                    "k": k,
                    "num_routes": len(routes),
                    "elapsed_ms_mean": mean_ms,
                    "elapsed_ms_stdev": std_ms,
                }
                rows.append(row)
                logger.info("device=%s N=%d K=%d -> %.2f ms", device, n, k, mean_ms)

    crossover_note = "GPU not faster than CPU in scanned grid"
    if "cuda" in devices:
        for k in K_GRID:
            for n in particle_grid:
                cpu = next(
                    (r for r in rows if r["device"] == "cpu" and r["k"] == k and r["particles"] == n),
                    None,
                )
                gpu = next(
                    (r for r in rows if r["device"] == "cuda" and r["k"] == k and r["particles"] == n),
                    None,
                )
                if cpu and gpu and gpu["elapsed_ms_mean"] < cpu["elapsed_ms_mean"]:
                    crossover_note = (
                        f"CUDA faster at N={n} K={k} "
                        f"({gpu['elapsed_ms_mean']:.1f}ms vs {cpu['elapsed_ms_mean']:.1f}ms)"
                    )
                    break
            else:
                continue
            break

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E2_multipool_scale_curve",
        "record_id": args.record_id,
        "particle_grid": particle_grid,
        "k_grid": K_GRID,
        "devices": devices,
        "crossover_note": crossover_note,
        "single_gpu_100k_k32_ms": next(
            (
                r["elapsed_ms_mean"]
                for r in rows
                if r["device"] == (devices[-1] if "cuda" in devices else "cpu")
                and r["particles"] == 100_000
                and r["k"] == 32
            ),
            None,
        ),
        "results": rows,
    }

    E2_DIR.mkdir(parents=True, exist_ok=True)
    json_path = E2_DIR / "multipool_scale_curve.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mid_n = particle_grid[len(particle_grid) // 2]
    _plot_particles(rows, FIG_DIR / "e2_scale_vs_particles.png", fixed_k=32)
    _plot_k(rows, FIG_DIR / "e2_scale_vs_k.png", fixed_n=mid_n)
    combined = FIG_DIR / "e2_scale_curve.png"
    _plot_particles(rows, combined, fixed_k=32)

    logger.info("JSON: %s", json_path)
    logger.info("Figure: %s", combined)
    logger.info("Crossover: %s", crossover_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
