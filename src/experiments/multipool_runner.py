"""多池两阶段 PSO/GA 实验运行器（Phase 6 E1/E2/E3 共用）。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from config.settings import PSO_CONFIG
from src.closed_form import global_closed_form_optimum
from src.cycle_finder import find_top_k_candidates
from src.models import MultiPoolCostModel, get_multipool_search_bounds
from src.optimizer import GAOptimizer, PSOOptimizer
from src.types import CandidateRoute, MultiPoolSnapshot


@dataclass
class SolverRunResult:
    solver: str
    best_fitness: float
    elapsed_ms: float
    converged_at_iter: int
    best_position: list[float]


def prepare_routes(
    snapshot: MultiPoolSnapshot,
    *,
    top_k: int = 32,
    min_depth_usd: float = 1.0,
) -> list[CandidateRoute]:
    return find_top_k_candidates(
        snapshot,
        top_k=top_k,
        min_effective_depth_usd=min_depth_usd,
    )


def closed_form_profit(
    snapshot: MultiPoolSnapshot,
    routes: list[CandidateRoute],
    *,
    latency_risk_lambda: float | None = None,
) -> tuple[float, float, CandidateRoute | None]:
    return global_closed_form_optimum(
        routes,
        snapshot,
        latency_risk_lambda=latency_risk_lambda,
    )


def _make_positions(num: int, bounds: torch.Tensor, device: str) -> torch.Tensor:
    low, high = bounds[:, 0], bounds[:, 1]
    return torch.rand(num, bounds.shape[0], device=device) * (high - low) + low


def run_pso_multipool(
    snapshot: MultiPoolSnapshot,
    routes: list[CandidateRoute],
    *,
    num_particles: int = 2000,
    max_iter: int = 100,
    seed: int = 42,
    device: str | None = None,
) -> SolverRunResult:
    device = device or PSO_CONFIG["device"]
    bounds = get_multipool_search_bounds(snapshot, len(routes), device)
    model = MultiPoolCostModel.from_routes(snapshot, routes)

    optimizer = PSOOptimizer(
        num_particles=num_particles,
        dim=bounds.shape[0],
        bounds=bounds,
        device=device,
        w=PSO_CONFIG["w"],
        c1=PSO_CONFIG["c1"],
        c2=PSO_CONFIG["c2"],
        seed=seed,
    )

    def fitness_fn(positions):
        return model.evaluate_fitness(positions, snapshot)

    result = optimizer.search(fitness_fn=fitness_fn, max_iter=max_iter)
    return SolverRunResult(
        solver="pso",
        best_fitness=result.best_fitness,
        elapsed_ms=result.elapsed_ms,
        converged_at_iter=result.converged_at_iter,
        best_position=result.best_position,
    )


def run_ga_multipool(
    snapshot: MultiPoolSnapshot,
    routes: list[CandidateRoute],
    *,
    pop_size: int = 2000,
    max_iter: int = 100,
    seed: int = 42,
    device: str | None = None,
) -> SolverRunResult:
    device = device or "cpu"
    bounds = get_multipool_search_bounds(snapshot, len(routes), device)
    model = MultiPoolCostModel.from_routes(snapshot, routes)

    optimizer = GAOptimizer(
        pop_size=pop_size,
        dim=bounds.shape[0],
        bounds=bounds,
        device=device,
        seed=seed,
    )

    def fitness_fn(positions):
        return model.evaluate_fitness(positions.to(device), snapshot)

    result = optimizer.search(fitness_fn=fitness_fn, max_iter=max_iter)
    return SolverRunResult(
        solver="ga",
        best_fitness=result.best_fitness,
        elapsed_ms=result.elapsed_ms,
        converged_at_iter=result.converged_at_iter,
        best_position=result.best_position,
    )


def timed_fitness_batch(
    snapshot: MultiPoolSnapshot,
    routes: list[CandidateRoute],
    *,
    num_particles: int,
    device: str = "cpu",
    repeats: int = 3,
    warmup: int = 3,
) -> tuple[float, float]:
    """批量 fitness [N,K] 墙钟（毫秒），返回 (mean, stdev)。"""
    model = MultiPoolCostModel.from_routes(snapshot, routes)
    bounds = get_multipool_search_bounds(snapshot, len(routes), device)
    positions = _make_positions(num_particles, bounds, device)

    for _ in range(warmup):
        model.evaluate_fitness(positions, snapshot)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.evaluate_fitness(positions, snapshot)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

    if len(samples) == 1:
        return samples[0], 0.0
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    return mean, var ** 0.5
