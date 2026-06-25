"""torch.distributed 初始化与辅助函数。"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def init_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """初始化进程组。未由 torchrun 启动时返回 (0, 0, 1)。"""
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not dist.is_initialized():
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except RuntimeError:
        return False


def device_for_rank(local_rank: int, world_size: int = 1) -> str:
    env_device = os.getenv("DEVICE", "").strip().lower()
    if env_device == "cpu":
        return "cpu"
    if world_size > 1 and cuda_usable():
        return f"cuda:{local_rank}"
    if cuda_usable() and local_rank == 0:
        return "cuda:0"
    return "cpu"
