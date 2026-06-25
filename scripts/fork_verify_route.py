"""CLI：Fork / QuoterV2 路由验真（Phase 5）。

示例:
    python scripts/fork_verify_route.py --snapshot data/snapshots/snap_latest.json
    python scripts/fork_verify_route.py --snapshot data/snapshots/snap_latest.json --route-idx 0 --amount 0.5
    python scripts/fork_verify_route.py --snapshot data/snapshots/snap_latest.json --no-anvil
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import SNAPSHOTS_DIR
from src.closed_form import optimize_route_amount
from src.cycle_finder import find_top_k_candidates
from src.fork_verify import AnvilProcess, append_fork_verify_result, verify_route
from src.models import MultiPoolCostModel
from src.snapshot_builder import load_multi_pool_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify routes via QuoterV2 / anvil fork")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=SNAPSHOTS_DIR / "snap_latest.json",
        help="Multi-pool snapshot JSON",
    )
    parser.add_argument(
        "--route-idx",
        type=int,
        default=None,
        help="Candidate route index in Top-K (default: 0)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=None,
        help="ETH amount to verify (default: closed-form optimal for route)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Max routes to verify (default 2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Stage-1 Top-K for candidate list",
    )
    parser.add_argument(
        "--swap-exec",
        action="store_true",
        help="L1 anvil fork: execute SwapRouter02 swap (state-changing); L2 still QuoterV2",
    )
    parser.add_argument(
        "--no-anvil",
        action="store_true",
        help="Skip anvil; use RPC QuoterV2 at pinned blocks only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default data/results/fork_verify.jsonl)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.snapshot.is_file():
        logger.error("Snapshot not found: %s", args.snapshot)
        return 1

    if args.swap_exec and args.no_anvil:
        logger.error("Cannot use --swap-exec with --no-anvil")
        return 1

    if not args.no_anvil and not args.swap_exec and not AnvilProcess.is_available():
        logger.info("anvil not found in PATH — using RPC QuoterV2 mode (see --no-anvil)")
    elif args.swap_exec and not AnvilProcess.is_available():
        logger.error("Foundry anvil required for --swap-exec. Install: https://getfoundry.sh")
        return 1

    snapshot = load_multi_pool_snapshot(args.snapshot)
    routes = find_top_k_candidates(snapshot, top_k=args.top_k, min_effective_depth_usd=1.0)
    if not routes:
        logger.error("No candidates from stage-1 filter")
        return 1

    if args.route_idx is not None:
        if args.route_idx < 0 or args.route_idx >= len(routes):
            logger.error("route-idx %s out of range [0, %s)", args.route_idx, len(routes))
            return 1
        selected = [routes[args.route_idx]]
    else:
        selected = routes[: max(args.limit, 1)]

    print(f"\n=== Fork / QuoterV2 Verify ===")
    print(f"snapshot: {snapshot.snapshot_id} | L1={snapshot.l1_block} L2={snapshot.l2_block}")
    print(f"routes to verify: {len(selected)}\n")

    model = MultiPoolCostModel.from_routes(snapshot, routes)
    exit_code = 0

    for route in selected:
        if args.amount is not None:
            amount = args.amount
        else:
            amount, _ = optimize_route_amount(route, snapshot, model)

        result = verify_route(
            snapshot,
            route,
            amount,
            use_anvil=not args.no_anvil,
            swap_exec=args.swap_exec,
            model=model,
        )
        append_fork_verify_result(result, args.out)

        status = "OK" if result.quoter_ok else "FAIL"
        print(f"[{status}] route #{route.route_id} {route.l1_pool_id} x {route.l2_pool_id}")
        print(f"  mode: {result.verification_mode} | amount: {result.amount_in_eth:.4f} ETH")
        print(f"  simulated_profit: ${result.simulated_profit:,.2f}")
        print(f"  fork_profit:      ${result.fork_profit:,.2f}")
        print(f"  rel_error:        {result.rel_error:.2%}")
        if result.notes:
            print(f"  notes: {result.notes}")
        print()

        if not result.quoter_ok:
            exit_code = 1

    out_path = args.out or (ROOT / "data" / "results" / "fork_verify.jsonl")
    print(f"Results appended to {out_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
