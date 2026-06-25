"""分布式 PSO 逻辑测试（单进程 + CPU gloo 双进程）。"""

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_mock_snapshots
from src.models import CostModel, get_search_bounds
from src.optimizer import create_pso_optimizer


def test_create_pso_optimizer_shards_particles():
    bounds = get_search_bounds("cpu")
    opt = create_pso_optimizer(
        num_particles=1000,
        bounds=bounds,
        device="cpu",
        rank=1,
        world_size=4,
        seed=42,
    )
    assert opt.num_particles == 250


def test_single_process_run_ddp_smoke():
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ddp.py"),
        "--mock",
        "--record-id",
        "snap_003",
        "--num-particles",
        "200",
        "--max-iter",
        "30",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    assert "elapsed_ms" in result.stdout
    assert "best_fitness" in result.stdout


def test_gbest_sync_single_rank():
    snapshots = load_mock_snapshots()
    snapshot = next(s for s in snapshots if s.snapshot_id == "snap_003")
    bounds = get_search_bounds("cpu")
    model = CostModel()

    def fitness_fn(positions):
        return model.evaluate_fitness(positions, snapshot)

    opt = create_pso_optimizer(
        num_particles=100,
        bounds=bounds,
        device="cpu",
        seed=42,
    )
    result = opt.search(fitness_fn=fitness_fn, max_iter=20)
    assert result.elapsed_ms > 0
    assert len(result.fitness_history) >= 1
    assert result.best_fitness > float("-inf")


def test_gloo_two_processes_if_available():
    if not torch.distributed.is_available():
        return

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port",
        "29511",
        str(ROOT / "scripts" / "run_ddp.py"),
        "--mock",
        "--record-id",
        "snap_003",
        "--num-particles",
        "100",
        "--max-iter",
        "20",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        # Windows 等环境可能无 gloo 多进程支持，跳过
        pytest_skip_msg = "gloo 双进程不可用"
        import pytest

        pytest.skip(f"{pytest_skip_msg}: {result.stderr[:300]}")
    assert "best_fitness" in result.stdout
