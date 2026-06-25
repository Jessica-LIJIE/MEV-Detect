"""两阶段框架 CLI：阶段1 候选环 + 阶段2 PSO 资金分配。

示例:
    python scripts/run_two_stage.py --snapshot data/snapshots/snap_latest.json --stage1-only
    python scripts/run_two_stage.py --snapshot data/snapshots/snap_latest.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import CYCLE_FINDER_CONFIG, PSO_CONFIG, SNAPSHOTS_DIR
from src.closed_form import closed_form_upper_bound
from src.cycle_finder import find_top_k_candidates, format_candidate_route
from src.models import (
    MultiPoolCostModel,
    build_multipool_strategy,
    get_multipool_search_bounds,
)
from src.optimizer import create_pso_optimizer
from src.snapshot_builder import load_multi_pool_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage cross-layer routing")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=SNAPSHOTS_DIR / "snap_latest.json",
        help="Multi-pool snapshot JSON path",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run stage-1 candidate filtering only",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Override Top-K (default {CYCLE_FINDER_CONFIG['top_k']})",
    )
    parser.add_argument(
        "--min-depth-usd",
        type=float,
        default=None,
        help="Override min effective depth USD filter",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=None,
        help="PSO particle count (default from PSO_CONFIG)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="PSO max iterations (default from PSO_CONFIG)",
    )
    parser.add_argument(
        "--stage1-mode",
        choices=("full", "mvp"),
        default=None,
        help=f"Stage-1 finder mode (default {CYCLE_FINDER_CONFIG.get('mode', 'full')})",
    )
    return parser.parse_args()


def run_stage1(
    snapshot_path: Path,
    *,
    top_k: int | None,
    min_depth_usd: float | None,
    stage1_mode: str | None = None,
) -> int:
    if not snapshot_path.is_file():
        logger.error("Snapshot not found: %s", snapshot_path)
        return 1

    snapshot = load_multi_pool_snapshot(snapshot_path)
    mode = stage1_mode or CYCLE_FINDER_CONFIG.get("mode", "full")
    routes = find_top_k_candidates(
        snapshot,
        top_k=top_k,
        min_effective_depth_usd=min_depth_usd,
        mode=mode,
    )

    print(f"\n=== Stage 1: Top-{len(routes)} candidates ({mode}) ===")
    print(f"snapshot_id: {snapshot.snapshot_id}")
    print(f"L1 block: {snapshot.l1_block} | L2 block: {snapshot.l2_block}")
    print(f"L1 pools: {snapshot.num_l1_pools} | L2 pools: {snapshot.num_l2_pools}")
    print(f"min_effective_depth_usd: {min_depth_usd or CYCLE_FINDER_CONFIG['min_effective_depth_usd']}")
    print()

    if not routes:
        print("No candidates passed depth filter.")
        return 0

    for route in routes:
        print(format_candidate_route(route, snapshot))

    print(f"\nBest score: {routes[0].score:,.2f} | Worst in Top-K: {routes[-1].score:,.2f}")
    return 0


def run_full_two_stage(
    snapshot_path: Path,
    *,
    top_k: int | None,
    min_depth_usd: float | None,
    num_particles: int | None,
    max_iter: int | None,
    stage1_mode: str | None = None,
) -> int:
    if not snapshot_path.is_file():
        logger.error("Snapshot not found: %s", snapshot_path)
        return 1

    snapshot = load_multi_pool_snapshot(snapshot_path)
    mode = stage1_mode or CYCLE_FINDER_CONFIG.get("mode", "full")
    routes = find_top_k_candidates(
        snapshot,
        top_k=top_k,
        min_effective_depth_usd=min_depth_usd,
        mode=mode,
    )
    if not routes:
        print("No candidates passed stage-1 filter; aborting PSO.")
        return 0

    print(f"\n=== Stage 1: {len(routes)} candidates (Top-K, {mode}) ===")
    print(f"Best static score: {routes[0].score:,.2f} | {routes[0].l1_pool_id} x {routes[0].l2_pool_id}")

    device = PSO_CONFIG["device"]
    bounds = get_multipool_search_bounds(snapshot, len(routes), device)
    model = MultiPoolCostModel.from_routes(snapshot, routes)
    particles = num_particles or PSO_CONFIG["num_particles"]
    iterations = max_iter or PSO_CONFIG["max_iter"]

    optimizer = create_pso_optimizer(
        num_particles=particles,
        bounds=bounds,
        device=device,
        w=PSO_CONFIG["w"],
        c1=PSO_CONFIG["c1"],
        c2=PSO_CONFIG["c2"],
        seed=42,
    )

    def fitness_fn(positions, snap=snapshot):
        return model.evaluate_fitness(positions, snap)

    print(f"\n=== Stage 2: PSO ({particles} particles, {iterations} iter, device={device}) ===")
    result = optimizer.search(fitness_fn=fitness_fn, max_iter=iterations)
    strategy = build_multipool_strategy(snapshot, routes, result)

    best_route = routes[int(min(max(int(float(result.best_position[1])), 0), len(routes) - 1))]
    closed = closed_form_upper_bound(best_route, snapshot, model=model)

    print(f"PSO best fitness: ${result.best_fitness:,.2f}")
    print(f"  amount_in: {strategy.amount_in_eth:.4f} ETH")
    print(
        f"  route: L1[{best_route.l1_pool_idx}] {best_route.l1_pool_id} "
        f"x L2[{best_route.l2_pool_idx}] {best_route.l2_pool_id} | bridge={best_route.bridge_id}"
    )
    print(f"  gas ${strategy.gas_cost_usd:.2f} | bridge ${strategy.bridge_fee_usd:.2f}")
    print(f"  elapsed: {result.elapsed_ms:.1f} ms | converged iter: {result.converged_at_iter}")
    print(
        f"Closed-form upper bound (same route): "
        f"${closed['net_profit_usd']:,.2f} @ {closed['amount_in_eth']:.4f} ETH"
    )
    print(strategy.summary())
    return 0


def main() -> int:
    args = _parse_args()

    if args.stage1_only:
        return run_stage1(
            args.snapshot,
            top_k=args.top_k,
            min_depth_usd=args.min_depth_usd,
            stage1_mode=args.stage1_mode,
        )

    return run_full_two_stage(
        args.snapshot,
        top_k=args.top_k,
        min_depth_usd=args.min_depth_usd,
        num_particles=args.particles,
        max_iter=args.max_iter,
        stage1_mode=args.stage1_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
