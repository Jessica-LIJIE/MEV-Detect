"""读取 live_log.jsonl 并转为面板可用的扁平记录。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIVE_LOG = ROOT / "data" / "results" / "live_log.jsonl"


def load_jsonl_records(path: Path | None = None) -> list[dict]:
    log_path = path or DEFAULT_LIVE_LOG
    if not log_path.exists():
        return []

    records: list[dict] = []
    with open(log_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{log_path}:{line_no} JSON 解析失败: {exc}") from exc
    return records


def flatten_record(record: dict) -> dict:
    strategy = record.get("strategy") or {}
    profile = record.get("pso_profile") or {}

    row = {
        "source": record.get("source"),
        "snapshot_id": record.get("snapshot_id"),
        "timestamp": record.get("timestamp"),
        "trigger": record.get("trigger"),
        "scenario": record.get("scenario"),
        "l1_price": record.get("l1_price"),
        "l2_price": record.get("l2_price"),
        "spread": record.get("spread"),
        "spread_pct": record.get("spread_pct"),
        "l2_lag_ms": record.get("l2_lag_ms"),
        "swap_amount_usd": record.get("swap_amount_usd"),
        "l1_block": record.get("l1_block"),
        "l2_block": record.get("l2_block"),
        "gas_l1_gwei": record.get("gas_l1_gwei"),
        "gas_l2_gwei": record.get("gas_l2_gwei"),
        "bridge_fee_usd": record.get("bridge_fee_usd"),
        "has_opportunity": record.get("has_opportunity"),
        "amount_in_eth": strategy.get("amount_in_eth"),
        "route_l1": strategy.get("route_l1"),
        "route_l2": strategy.get("route_l2"),
        "bridge_path": strategy.get("bridge_path"),
        "expected_profit_usd": strategy.get("expected_profit_usd"),
        "gas_cost_usd": strategy.get("gas_cost_usd"),
        "strategy_bridge_fee_usd": strategy.get("bridge_fee_usd"),
        "search_elapsed_ms": strategy.get("search_elapsed_ms"),
        "pso_profile_name": profile.get("name"),
        "pso_particles": profile.get("num_particles"),
        "pso_max_iter": profile.get("max_iter"),
        "pso_reason": profile.get("reason"),
    }
    return row


def records_to_dataframe(
    records: list[dict],
    *,
    dedupe: bool = True,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    rows = [flatten_record(r) for r in records]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp", kind="stable")

    if dedupe and not df.empty:
        df = df.drop_duplicates(subset=["snapshot_id"], keep="last")

    return df.reset_index(drop=True)


def load_detection_dataframe(
    path: Path | None = None,
    *,
    source: str | None = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    records = load_jsonl_records(path)
    if source and source != "all":
        records = [r for r in records if r.get("source") == source]
    return records_to_dataframe(records, dedupe=dedupe)
