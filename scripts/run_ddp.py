"""多 GPU PSO 入口（E2 实验）。

单进程（本地调试）:
    python scripts/run_ddp.py --mock --record-id snap_003

多卡（云主机）:
    torchrun --nproc_per_node=2 scripts/run_ddp.py --mock --record-id snap_003
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

from config.settings import PSO_CONFIG
from src.data_loader import load_mock_snapshots
from src.distributed_utils import destroy_distributed, device_for_rank, init_distributed
from src.models import CostModel, build_strategy, get_search_bounds
from src.optimizer import create_pso_optimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RECORD_ID = "snap_003"
RESULTS_DIR = ROOT / "data" / "results"


def _load_snapshot(record_id: str):
    snapshots = load_mock_snapshots()
    matched = [s for s in snapshots if s.snapshot_id == record_id]
    if not matched:
        raise ValueError(f"未找到快照: {record_id}")
    return matched[0]


def run_search(
    snapshot,
    *,
    num_particles: int,
    max_iter: int,
    seed: int,
    rank: int,
    local_rank: int,
    world_size: int,
) -> dict:
    device = device_for_rank(local_rank, world_size)
    bounds = get_search_bounds(device)
    cost_model = CostModel()

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
        return cost_model.evaluate_fitness(positions, snapshot)

    if rank == 0:
        logger.info(
            "PSO 搜索 | 快照=%s | 总粒子=%d | 每 rank=%d | world_size=%d | device=%s",
            snapshot.snapshot_id,
            num_particles,
            optimizer.num_particles,
            world_size,
            device,
        )

    result = optimizer.search(fitness_fn=fitness_fn, max_iter=max_iter)
    strategy = build_strategy(snapshot, result)

    return {
        "snapshot_id": snapshot.snapshot_id,
        "num_particles_total": num_particles,
        "num_particles_per_rank": optimizer.num_particles,
        "world_size": world_size,
        "max_iter": max_iter,
        "seed": seed,
        "device": device,
        "elapsed_ms": result.elapsed_ms,
        "converged_at_iter": result.converged_at_iter,
        "best_fitness": result.best_fitness,
        "best_position": result.best_position,
        "expected_profit_usd": strategy.expected_profit_usd,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="多 GPU PSO 搜索（E2）")
    parser.add_argument("--mock", action="store_true", help="使用 mock_mempool.json")
    parser.add_argument("--record-id", type=str, default=DEFAULT_RECORD_ID)
    parser.add_argument("--num-particles", type=int, default=PSO_CONFIG["num_particles"])
    parser.add_argument("--max-iter", type=int, default=PSO_CONFIG["max_iter"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-output",
        type=str,
        default="",
        help="将结果写入 JSON 文件（仅 rank 0）",
    )
    args = parser.parse_args()

    if not args.mock:
        parser.error("当前仅支持 --mock 模式")

    rank, local_rank, world_size = init_distributed()

    try:
        snapshot = _load_snapshot(args.record_id)
        payload = run_search(
            snapshot,
            num_particles=args.num_particles,
            max_iter=args.max_iter,
            seed=args.seed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

        if rank == 0:
            logger.info(
                "完成 | elapsed=%.2f ms | fitness=%.2f | profit=$%.2f | iter=%d",
                payload["elapsed_ms"],
                payload["best_fitness"],
                payload["expected_profit_usd"],
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
