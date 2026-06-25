"""多 GPU 多池 PSO（E2 多卡实验）。

单进程冒烟:
    python scripts/run_ddp_multipool.py --record-id mock_mp_001 --num-particles 1000 --max-iter 80

云主机 2 卡:
    torchrun --nproc_per_node=2 scripts/run_ddp_multipool.py --record-id mock_mp_001 \\
        --num-particles 8000 --max-iter 80 --json-output data/results/ddp_mp_2gpu.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from config.settings import MOCK_MULTIPOOL_DIR, PSO_CONFIG
from src.distributed_utils import destroy_distributed, device_for_rank, init_distributed
from src.experiments.multipool_runner import prepare_routes
from src.models import MultiPoolCostModel, get_multipool_search_bounds
from src.multipool_mock import load_mock_multipool_records
from src.optimizer import create_pso_optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_snapshot(record_id: str):
    records = load_mock_multipool_records(MOCK_MULTIPOOL_DIR / "records.json")
    for item in records:
        if item.snapshot.snapshot_id == record_id:
            return item.snapshot
    raise ValueError(f"未找到 mock 快照: {record_id}")


def run_multipool_pso(
    snapshot,
    routes,
    *,
    num_particles: int,
    max_iter: int,
    seed: int,
    rank: int,
    local_rank: int,
    world_size: int,
) -> dict:
    device = device_for_rank(local_rank, world_size)
    bounds = get_multipool_search_bounds(snapshot, len(routes), device)
    model = MultiPoolCostModel.from_routes(snapshot, routes)

    optimizer = create_pso_optimizer(
        num_particles=num_particles,
        bounds=bounds,
        device=device,
        rank=rank,
        world_size=world_size,
        w=PSO_CONFIG["w"],
        c1=PSO_CONFIG["c1"],
        c2=PSO_CONFIG["c2"],
        seed=seed,
    )

    def fitness_fn(positions):
        return model.evaluate_fitness(positions, snapshot)

    if rank == 0:
        logger.info(
            "多池 PSO | snapshot=%s | K=%d | 总粒子=%d | 每 rank=%d | world=%d | device=%s",
            snapshot.snapshot_id,
            len(routes),
            num_particles,
            optimizer.num_particles,
            world_size,
            device,
        )

    result = optimizer.search(fitness_fn=fitness_fn, max_iter=max_iter)

    return {
        "snapshot_id": snapshot.snapshot_id,
        "record_id": snapshot.snapshot_id,
        "num_candidates": len(routes),
        "num_particles_total": num_particles,
        "num_particles_per_rank": optimizer.num_particles,
        "world_size": world_size,
        "max_iter": max_iter,
        "seed": seed,
        "device": device,
        "elapsed_ms": result.elapsed_ms,
        "converged_at_iter": result.converged_at_iter,
        "best_fitness": result.best_fitness,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="多 GPU 多池 PSO（E2）")
    parser.add_argument("--record-id", type=str, default="mock_mp_001")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--num-particles", type=int, default=PSO_CONFIG["num_particles"])
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=str, default="")
    args = parser.parse_args()

    rank, local_rank, world_size = init_distributed()

    try:
        snapshot = _load_snapshot(args.record_id)
        routes = prepare_routes(snapshot, top_k=args.top_k)
        payload = run_multipool_pso(
            snapshot,
            routes,
            num_particles=args.num_particles,
            max_iter=args.max_iter,
            seed=args.seed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

        if rank == 0:
            logger.info(
                "完成 | elapsed=%.2f ms | fitness=%.2f | iter=%d",
                payload["elapsed_ms"],
                payload["best_fitness"],
                payload["converged_at_iter"],
            )
            if args.json_output:
                out_path = Path(args.json_output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info("结果已写入 %s", out_path)
            else:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
