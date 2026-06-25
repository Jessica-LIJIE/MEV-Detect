"""log_loader 单元测试。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.log_loader import load_detection_dataframe, load_jsonl_records


def test_load_jsonl_not_empty():
    records = load_jsonl_records()
    assert len(records) > 0


def test_dataframe_dedupe():
    df = load_detection_dataframe(dedupe=True)
    assert df["snapshot_id"].is_unique


def test_flatten_strategy_fields():
    df = load_detection_dataframe(dedupe=True)
    assert "expected_profit_usd" in df.columns
    assert "search_elapsed_ms" in df.columns
