import time
from typing import Callable

import torch
import torch.distributed as dist

from src.distributed_utils import is_distributed
from src.types import SearchResult


class PSOOptimizer:
    """完全向量化的粒子群优化器，粒子级更新无 Python for 循环。"""

    def __init__(
        self,
        num_particles: int,
        dim: int,
        bounds: torch.Tensor,
        device: str = "cpu",
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int | None = None,
    ):
        self.num_particles = num_particles
        self.dim = dim
        self.device = device
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds.to(device)

        if seed is not None:
            torch.manual_seed(seed)

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.positions = torch.rand(num_particles, dim, device=device) * (high - low) + low
        self.velocities = torch.zeros(num_particles, dim, device=device)
        self.pbest_positions = self.positions.clone()
        self.pbest_fitness = torch.full((num_particles,), float("-inf"), device=device)
        self.gbest_position = self.positions[0].clone()
        self.gbest_fitness = float("-inf")

    def _clamp_positions(self) -> None:
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.positions = torch.max(self.positions, low)
        self.positions = torch.min(self.positions, high)

    def _sync_gbest(
        self,
        fitness: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """单卡返回本地最优；多卡 all_gather 取全局最优 fitness 并 broadcast 对应位置。"""
        best_idx = fitness.argmax()
        local_best_fit = fitness[best_idx].detach().clone()
        local_best_pos = positions[best_idx].detach().clone()

        if not is_distributed():
            return local_best_fit.item(), local_best_pos

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        fit_payload = local_best_fit.reshape(1).to(device=self.device, dtype=torch.float32)
        fit_buffers = [
            torch.zeros(1, device=self.device, dtype=torch.float32) for _ in range(world_size)
        ]
        dist.all_gather(fit_buffers, fit_payload)

        fits = torch.cat(fit_buffers)
        winner_rank = int(fits.argmax().item())
        global_fit = float(fits[winner_rank].item())

        pos_payload = (
            local_best_pos
            if rank == winner_rank
            else torch.zeros(self.dim, device=self.device, dtype=positions.dtype)
        )
        dist.broadcast(pos_payload, src=winner_rank)

        return global_fit, pos_payload.clone()

    def search(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        max_iter: int = 100,
        patience: int = 15,
    ) -> SearchResult:
        start = time.perf_counter()
        history: list[float] = []

        fitness = fitness_fn(self.positions)
        self.pbest_fitness = fitness.clone()
        self.pbest_positions = self.positions.clone()

        self.gbest_fitness, self.gbest_position = self._sync_gbest(fitness, self.positions)
        history.append(self.gbest_fitness)

        no_improve = 0
        converged_at = max_iter

        for iteration in range(max_iter):
            r1 = torch.rand(self.num_particles, self.dim, device=self.device)
            r2 = torch.rand(self.num_particles, self.dim, device=self.device)

            self.velocities = (
                self.w * self.velocities
                + self.c1 * r1 * (self.pbest_positions - self.positions)
                + self.c2 * r2 * (self.gbest_position.unsqueeze(0) - self.positions)
            )
            self.positions = self.positions + self.velocities
            self._clamp_positions()

            fitness = fitness_fn(self.positions)

            improved = fitness > self.pbest_fitness
            self.pbest_fitness = torch.where(improved, fitness, self.pbest_fitness)
            self.pbest_positions = torch.where(
                improved.unsqueeze(1),
                self.positions,
                self.pbest_positions,
            )

            global_fit, global_pos = self._sync_gbest(fitness, self.positions)
            history.append(global_fit)

            if global_fit > self.gbest_fitness + 1e-6:
                self.gbest_fitness = global_fit
                self.gbest_position = global_pos
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience and iteration >= 20:
                converged_at = iteration + 1
                break

        if is_distributed():
            dist.barrier()

        elapsed_ms = (time.perf_counter() - start) * 1000
        return SearchResult(
            best_position=self.gbest_position.cpu().tolist(),
            best_fitness=self.gbest_fitness,
            converged_at_iter=converged_at,
            elapsed_ms=elapsed_ms,
            fitness_history=history,
        )


class GAOptimizer:
    """对照组：传统遗传算法，个体评估与遗传操作使用串行循环。"""

    def __init__(
        self,
        pop_size: int,
        dim: int,
        bounds: torch.Tensor,
        device: str = "cpu",
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        seed: int | None = None,
    ):
        self.pop_size = pop_size
        self.dim = dim
        self.device = device
        self.bounds = bounds.cpu()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        if seed is not None:
            torch.manual_seed(seed)

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.population = torch.rand(pop_size, dim) * (high - low) + low

    def _evaluate_serial(
        self,
        population: torch.Tensor,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        fitness_list = []
        for i in range(population.shape[0]):
            single = population[i : i + 1].to(self.device)
            fitness_list.append(fitness_fn(single).squeeze())
        return torch.stack(fitness_list)

    def _tournament_select(self, fitness: torch.Tensor, k: int = 3) -> torch.Tensor:
        indices = torch.randint(0, self.pop_size, (k,))
        winner = indices[fitness[indices].argmax()]
        return self.population[winner].clone()

    def search(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        max_iter: int = 100,
        patience: int = 15,
    ) -> SearchResult:
        start = time.perf_counter()
        history: list[float] = []
        best_fitness = float("-inf")
        best_individual = self.population[0].clone()
        no_improve = 0
        converged_at = max_iter

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        for generation in range(max_iter):
            fitness = self._evaluate_serial(self.population, fitness_fn)

            gen_best_idx = fitness.argmax()
            gen_best = fitness[gen_best_idx].item()
            history.append(gen_best)

            if gen_best > best_fitness:
                best_fitness = gen_best
                best_individual = self.population[gen_best_idx].clone()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience and generation >= 20:
                converged_at = generation + 1
                break

            new_population = []
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_select(fitness)
                parent2 = self._tournament_select(fitness)

                if torch.rand(1).item() < self.crossover_rate:
                    point = torch.randint(1, self.dim, (1,)).item()
                    child1 = torch.cat([parent1[:point], parent2[point:]])
                    child2 = torch.cat([parent2[:point], parent1[point:]])
                else:
                    child1, child2 = parent1.clone(), parent2.clone()

                for child in (child1, child2):
                    for d in range(self.dim):
                        if torch.rand(1).item() < self.mutation_rate:
                            child[d] = low[d] + torch.rand(1).item() * (high[d] - low[d])
                    child = torch.max(child, low)
                    child = torch.min(child, high)
                    new_population.append(child)
                    if len(new_population) >= self.pop_size:
                        break

            self.population = torch.stack(new_population[: self.pop_size])

        elapsed_ms = (time.perf_counter() - start) * 1000
        return SearchResult(
            best_position=best_individual.tolist(),
            best_fitness=best_fitness,
            converged_at_iter=converged_at,
            elapsed_ms=elapsed_ms,
            fitness_history=history,
        )


def create_pso_optimizer(
    num_particles: int,
    bounds: torch.Tensor,
    device: str = "cpu",
    rank: int = 0,
    world_size: int = 1,
    **kwargs,
) -> PSOOptimizer:
    """多 GPU 时按 rank 划分粒子子群；各 rank 使用 seed+rank 独立初始化。"""
    particles = num_particles
    if world_size > 1:
        particles = num_particles // world_size
        if particles <= 0:
            particles = num_particles

    seed = kwargs.pop("seed", None)
    if seed is not None and world_size > 1:
        seed = seed + rank

    return PSOOptimizer(
        num_particles=particles,
        dim=bounds.shape[0],
        bounds=bounds,
        device=device,
        seed=seed,
        **kwargs,
    )
