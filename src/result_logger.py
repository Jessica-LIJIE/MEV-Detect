"""将检测快照与 PSO 策略追加写入 JSONL，供 E4 可视化读取。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.types import ArbitrageStrategy, MarketSnapshot

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIVE_LOG = ROOT / "data" / "results" / "live_log.jsonl"


def snapshot_strategy_to_record(
    snapshot: MarketSnapshot,
    strategy: ArbitrageStrategy,
    *,
    source: str = "live",
    pso_profile: dict | None = None,
) -> dict:
    l1_price = snapshot.l1.eth_usdc_price
    l2_price = snapshot.l2.eth_usdc_price
    spread = l1_price - l2_price
    spread_pct = (spread / l2_price * 100) if l2_price else 0.0

    record = {
        "source": source,
        "snapshot_id": snapshot.snapshot_id,
        "timestamp": snapshot.timestamp,
        "trigger": snapshot.trigger,
        "scenario": snapshot.scenario,
        "l1_price": l1_price,
        "l2_price": l2_price,
        "spread": spread,
        "spread_pct": round(spread_pct, 4),
        "l2_lag_ms": snapshot.l2_lag_ms,
        "swap_amount_usd": snapshot.swap_amount_usd,
        "l1_block": snapshot.l1.block_number,
        "l2_block": snapshot.l2.block_number,
        "gas_l1_gwei": snapshot.gas.l1_gwei,
        "gas_l2_gwei": snapshot.gas.l2_gwei,
        "bridge_fee_usd": snapshot.bridge_fee_usd,
        "strategy": asdict(strategy),
        "has_opportunity": strategy.expected_profit_usd > 0,
    }
    if pso_profile is not None:
        record["pso_profile"] = pso_profile
    return record


def append_result_log(
    snapshot: MarketSnapshot,
    strategy: ArbitrageStrategy,
    path: Path | None = None,
    *,
    source: str = "live",
    pso_profile: dict | None = None,
) -> None:
    log_path = path or DEFAULT_LIVE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = snapshot_strategy_to_record(
        snapshot, strategy, source=source, pso_profile=pso_profile
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
