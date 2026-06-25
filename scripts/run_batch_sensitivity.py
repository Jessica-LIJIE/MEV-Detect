"""±1 batch 敏感性：固定 L2 钉块，L1 池状态在 l1_batch±1 与原始钉块间对比 PSO/Quoter。

示例:
    python scripts/run_batch_sensitivity.py
    python scripts/run_batch_sensitivity.py --files snap_batch_01.json snap_batch_04.json
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
from src.experiments.multipool_runner import run_pso_multipool
from src.fork_verify import verify_route
from src.snapshot_builder import load_multi_pool_snapshot, rebuild_snapshot_with_l1_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = ROOT / "data" / "results" / "batch_sensitivity.json"

VARIANT_ORDER = ("pinned", "batch-1", "batch", "batch+1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L1 batch ±1 sensitivity on pinned L2 snapshots")
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR)
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Snapshot filenames under snapshots-dir (default: all *.json, skip idempotent_b)",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--quoter-amount", type=float, default=0.01, help="USDC route verify ETH amount")
    parser.add_argument("--skip-quoter", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def _snapshot_paths(args: argparse.Namespace) -> list[Path]:
    if args.files:
        return [args.snapshots_dir / name for name in args.files]
    paths = sorted(args.snapshots_dir.glob("*.json"))
    return [p for p in paths if p.name != "snap_idempotent_b.json"]


def _usdc_route(routes: list) -> object | None:
    for route in routes:
        if "usdc" in route.l1_pool_id.lower() and "usdc" in route.l2_pool_id.lower():
            return route
    return routes[0] if routes else None


def _variant_blocks(base_l1: int, l1_batch: int) -> list[tuple[str, int]]:
    return [
        ("pinned", base_l1),
        ("batch-1", l1_batch - 1),
        ("batch", l1_batch),
        ("batch+1", l1_batch + 1),
    ]


def _evaluate_variant(
    snapshot,
    *,
    top_k: int,
    particles: int,
    max_iter: int,
    quoter_amount: float,
    skip_quoter: bool,
) -> dict:
    routes = find_top_k_candidates(snapshot, top_k=top_k, min_effective_depth_usd=1.0)
    if not routes:
        return {
            "num_candidates": 0,
            "closed_form_profit": None,
            "pso_best_fitness": None,
            "has_positive_pso": False,
            "usdc_rel_error": None,
            "usdc_quoter_ok": None,
        }

    closed_profit, _, _ = global_closed_form_optimum(routes, snapshot)
    pso = run_pso_multipool(
        snapshot,
        routes,
        num_particles=particles,
        max_iter=max_iter,
        seed=42,
        device=PSO_CONFIG["device"],
    )

    usdc_rel_error = None
    usdc_quoter_ok = None
    if not skip_quoter:
        usdc = _usdc_route(routes)
        if usdc is not None:
            result = verify_route(snapshot, usdc, quoter_amount, use_anvil=False)
            usdc_rel_error = result.rel_error if result.quoter_ok else None
            usdc_quoter_ok = result.quoter_ok

    return {
        "num_candidates": len(routes),
        "closed_form_profit": closed_profit,
        "pso_best_fitness": pso.best_fitness,
        "pso_elapsed_ms": pso.elapsed_ms,
        "has_positive_pso": pso.best_fitness > 0,
        "usdc_rel_error": usdc_rel_error,
        "usdc_quoter_ok": usdc_quoter_ok,
    }


def _analyze_rows(rows: list[dict]) -> dict:
    batch_rows = [r for r in rows if r["variant"] in ("batch-1", "batch", "batch+1")]
    pso_vals = [r["pso_best_fitness"] for r in batch_rows if r.get("pso_best_fitness") is not None]
    positives = [r["has_positive_pso"] for r in batch_rows]
    analysis: dict = {
        "batch_pso_min": min(pso_vals) if pso_vals else None,
        "batch_pso_max": max(pso_vals) if pso_vals else None,
        "batch_pso_spread": (max(pso_vals) - min(pso_vals)) if len(pso_vals) >= 2 else 0.0,
        "batch_sign_flip": len(set(positives)) > 1 if positives else False,
    }
    pinned = next((r for r in rows if r["variant"] == "pinned"), None)
    batch0 = next((r for r in rows if r["variant"] == "batch"), None)
    if pinned and batch0 and pinned.get("pso_best_fitness") is not None and batch0.get("pso_best_fitness") is not None:
        analysis["pinned_vs_batch0_delta"] = batch0["pso_best_fitness"] - pinned["pso_best_fitness"]
    return analysis


def main() -> int:
    args = _parse_args()
    paths = _snapshot_paths(args)
    if not paths:
        logger.error("No snapshot files found")
        return 1

    per_snapshot: list[dict] = []

    print(f"\n=== ±1 Batch Sensitivity ({len(paths)} snapshots) ===\n")

    for path in paths:
        if not path.is_file():
            logger.warning("Skip missing file: %s", path)
            continue

        base = load_multi_pool_snapshot(path)
        if base.l1_batch is None:
            logger.warning("Skip %s: l1_batch is null", path.name)
            continue

        rows: list[dict] = []
        print(f"{path.name} | L2={base.l2_block} l1_batch={base.l1_batch} pinned_l1={base.l1_block}")

        for variant, l1_block in _variant_blocks(base.l1_block, base.l1_batch):
            if variant == "pinned":
                snap = base
            else:
                logger.info("  %s: refetch L1 at block %s", variant, l1_block)
                snap = rebuild_snapshot_with_l1_block(base, l1_block, fetch_primary_ticks=False)

            metrics = _evaluate_variant(
                snap,
                top_k=args.top_k,
                particles=args.particles,
                max_iter=args.max_iter,
                quoter_amount=args.quoter_amount,
                skip_quoter=args.skip_quoter,
            )
            row = {
                "variant": variant,
                "l1_block": l1_block,
                "batch_offset": {
                    "pinned": None,
                    "batch-1": -1,
                    "batch": 0,
                    "batch+1": 1,
                }[variant],
                **metrics,
            }
            rows.append(row)

            pso_s = f"${metrics['pso_best_fitness']:.2f}" if metrics["pso_best_fitness"] is not None else "n/a"
            rel_s = (
                f"{metrics['usdc_rel_error']:.2%}"
                if metrics.get("usdc_rel_error") is not None
                else ("fail" if metrics.get("usdc_quoter_ok") is False else "—")
            )
            print(f"  {variant:8s} L1={l1_block} PSO={pso_s} pos={metrics['has_positive_pso']} USDC rel={rel_s}")

        analysis = _analyze_rows(rows)
        entry = {
            "file": str(path.relative_to(ROOT)),
            "snapshot_id": base.snapshot_id,
            "l2_block": base.l2_block,
            "l1_batch": base.l1_batch,
            "pinned_l1_block": base.l1_block,
            "rows": rows,
            "analysis": analysis,
        }
        per_snapshot.append(entry)
        print(
            f"  → batch PSO spread ${analysis['batch_pso_spread']:.2f}, "
            f"sign_flip={analysis['batch_sign_flip']}\n"
        )

    sign_flips = sum(1 for e in per_snapshot if e["analysis"]["batch_sign_flip"])
    max_spread = max((e["analysis"]["batch_pso_spread"] or 0.0) for e in per_snapshot) if per_snapshot else 0.0

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "batch_sensitivity",
        "description": "Fix L2 block; vary L1 pool state at pinned l1_block vs l1_batch±1",
        "variants": list(VARIANT_ORDER),
        "pso_config": {
            "particles": args.particles,
            "max_iter": args.max_iter,
            "top_k": args.top_k,
            "device": PSO_CONFIG["device"],
        },
        "quoter_amount_eth": args.quoter_amount,
        "per_snapshot": per_snapshot,
        "summary": {
            "snapshot_count": len(per_snapshot),
            "sign_flip_count": sign_flips,
            "max_batch_pso_spread_usd": max_spread,
            "notes": [
                "pinned = original snapshot l1_block used in E3 collection",
                "batch/batch±1 = L1 pools refetched at l1_batch ancestor block ±1",
                "USDC rel_error at fixed small amount; detection profit may differ at large size",
            ],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Summary: sign_flips={sign_flips}/{len(per_snapshot)}, max_spread=${max_spread:.2f}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
