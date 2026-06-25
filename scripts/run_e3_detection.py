"""E3：真实快照检测汇总 + fork 验真引用。

示例:
    python scripts/run_e3_detection.py
    python scripts/run_e3_detection.py --snapshots-dir data/snapshots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PSO_CONFIG, SNAPSHOTS_DIR
from src.closed_form import global_closed_form_optimum
from src.cycle_finder import find_top_k_candidates
from src.experiments.multipool_runner import prepare_routes, run_pso_multipool
from src.fork_verify import FORK_VERIFY_LOG, verify_route
from src.models import MultiPoolCostModel
from src.snapshot_builder import load_multi_pool_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3 multipool detection summary")
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--fork-verify", action="store_true", help="Run QuoterV2 verify on best route")
    parser.add_argument("--fork-amount", type=float, default=0.01)
    return parser.parse_args()


def _load_fork_results() -> list[dict]:
    if not FORK_VERIFY_LOG.is_file():
        return []
    rows = []
    with open(FORK_VERIFY_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = _parse_args()
    snap_paths = sorted(args.snapshots_dir.glob("*.json"))
    if not snap_paths:
        logger.error("No snapshots in %s", args.snapshots_dir)
        return 1

    per_snapshot: list[dict] = []
    positive_pso_count = 0

    print(f"\n=== E3 Detection Summary ({len(snap_paths)} snapshots) ===\n")

    for path in snap_paths:
        snapshot = load_multi_pool_snapshot(path)
        routes = prepare_routes(snapshot, top_k=args.top_k)
        closed_profit, closed_amount, closed_route = global_closed_form_optimum(routes, snapshot)

        pso = run_pso_multipool(
            snapshot,
            routes,
            num_particles=args.particles,
            max_iter=args.max_iter,
            seed=42,
            device=PSO_CONFIG["device"],
        )
        has_positive = pso.best_fitness > 0
        if has_positive:
            positive_pso_count += 1

        entry = {
            "snapshot_id": snapshot.snapshot_id,
            "file": str(path.relative_to(ROOT)),
            "l1_block": snapshot.l1_block,
            "l2_block": snapshot.l2_block,
            "l1_batch": snapshot.l1_batch,
            "num_candidates": len(routes),
            "closed_form_profit": closed_profit,
            "closed_form_amount_eth": closed_amount,
            "pso_best_fitness": pso.best_fitness,
            "pso_elapsed_ms": pso.elapsed_ms,
            "has_positive_pso": has_positive,
            "best_closed_route": (
                {
                    "l1_pool_idx": closed_route.l1_pool_idx,
                    "l2_pool_idx": closed_route.l2_pool_idx,
                    "bridge_id": closed_route.bridge_id,
                }
                if closed_route
                else None
            ),
        }
        per_snapshot.append(entry)

        print(
            f"{snapshot.snapshot_id}: candidates={len(routes)} "
            f"PSO=${pso.best_fitness:.2f} ({pso.elapsed_ms:.1f}ms) "
            f"closed=${closed_profit:.2f} "
            f"{'POSITIVE' if has_positive else 'no profit'}"
        )

    fork_rows = _load_fork_results()
    fork_for_snapshots = {
        r["snapshot_id"]: r for r in fork_rows
    }

    fork_verify_runs: list[dict] = []
    if args.fork_verify and per_snapshot:
        target = per_snapshot[0]
        snapshot = load_multi_pool_snapshot(ROOT / target["file"])
        routes = prepare_routes(snapshot, top_k=args.top_k)
        route = routes[0]
        for r in routes:
            if "usdc" in r.l1_pool_id.lower() and "usdc" in r.l2_pool_id.lower():
                route = r
                break
        result = verify_route(snapshot, route, args.fork_amount, use_anvil=False)
        fork_verify_runs.append(result.to_dict())
        print(
            f"\nFork verify: {result.snapshot_id} rel_error={result.rel_error:.2%} "
            f"sim=${result.simulated_profit:.2f} fork=${result.fork_profit:.2f}"
        )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "E3_multipool_detection",
        "snapshots_dir": str(args.snapshots_dir.relative_to(ROOT)),
        "snapshot_count": len(per_snapshot),
        "positive_pso_count": positive_pso_count,
        "pso_config": {
            "particles": args.particles,
            "max_iter": args.max_iter,
            "device": PSO_CONFIG["device"],
        },
        "per_snapshot": per_snapshot,
        "fork_verify_log_entries": len(fork_rows),
        "fork_verify_by_snapshot": fork_for_snapshots,
        "fork_verify_new_runs": fork_verify_runs,
        "notes": [
            "Mock multipool ablation results are NOT included (see data/mock_multipool/).",
            "Positive profit on live snapshots is rare when spread < gas+bridge.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "e3_detection_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nPositive PSO: {positive_pso_count}/{len(per_snapshot)}")
    print(f"Summary JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
