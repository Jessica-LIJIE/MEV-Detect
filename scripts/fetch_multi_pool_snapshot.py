"""CLI：从 RPC 拉取跨层多池快照并写入 JSON。

示例:
    python scripts/fetch_multi_pool_snapshot.py \\
        --l1-block 20100100 \\
        --l2-block 220001500 \\
        --out data/snapshots/snap_real_001.json

    python scripts/fetch_multi_pool_snapshot.py --latest --out data/snapshots/snap_latest.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import RPC, SNAPSHOTS_DIR
from src.rpc_utils import create_sync_w3
from src.snapshot_builder import build_multi_pool_snapshot, save_multi_pool_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch L1/L2 multi-pool snapshot via RPC")
    parser.add_argument("--l1-block", type=int, help="Ethereum block number (pinned)")
    parser.add_argument("--l2-block", type=int, help="Arbitrum block number (pinned)")
    parser.add_argument("--latest", action="store_true", help="Use latest block on both chains")
    parser.add_argument(
        "--out",
        type=Path,
        default=SNAPSHOTS_DIR / "snap_real_001.json",
        help="Output JSON path",
    )
    parser.add_argument("--snapshot-id", type=str, default=None, help="Custom snapshot_id")
    parser.add_argument(
        "--no-primary-ticks",
        action="store_true",
        help="Skip tick() fetch on primary pools",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not RPC["ethereum"]["http"] or not RPC["arbitrum"]["http"]:
        logger.error("ETH_HTTP_URL / ARB_HTTP_URL 未配置，请先填写 .env")
        return 1

    w3_l1 = create_sync_w3(RPC["ethereum"]["http"])
    w3_l2 = create_sync_w3(RPC["arbitrum"]["http"])

    if args.latest:
        l1_block = int(w3_l1.eth.block_number)
        l2_block = int(w3_l2.eth.block_number)
        logger.info("Using latest blocks: L1=%s L2=%s", l1_block, l2_block)
    elif args.l1_block is not None and args.l2_block is not None:
        l1_block = args.l1_block
        l2_block = args.l2_block
    else:
        logger.error("Specify --l1-block and --l2-block, or use --latest")
        return 1

    logger.info("Fetching multi-pool snapshot at L1=%s L2=%s ...", l1_block, l2_block)
    snapshot = build_multi_pool_snapshot(
        l1_block=l1_block,
        l2_block=l2_block,
        snapshot_id=args.snapshot_id,
        w3_l1=w3_l1,
        w3_l2=w3_l2,
        fetch_primary_ticks=not args.no_primary_ticks,
    )

    save_multi_pool_snapshot(snapshot, args.out)
    logger.info(
        "Saved %s | L1 pools=%d L2 pools=%d | l1_batch=%s | primary ticks L1=%s L2=%s",
        args.out,
        snapshot.num_l1_pools,
        snapshot.num_l2_pools,
        snapshot.l1_batch,
        next(p.ticks_loaded for p in snapshot.l1_pools if p.pool_id == "eth_usdc_005"),
        next(p.ticks_loaded for p in snapshot.l2_pools if p.pool_id == "arb_usdc_005"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
