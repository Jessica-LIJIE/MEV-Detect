"""校验 data/mock_mempool.json 格式。用法: python scripts/validate_mock_data.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOCK_FILE = ROOT / "data" / "mock_mempool.json"

REQUIRED_RECORD_FIELDS = ("id", "timestamp", "trigger", "l1", "l2", "l2_lag_ms", "gas", "bridge_fee_usd")
REQUIRED_POOL_FIELDS = ("chain", "block_number", "block_time", "pool_address", "sqrt_price_x96", "liquidity", "tick", "eth_usdc_price")
REQUIRED_GAS_FIELDS = ("l1_gwei", "l2_gwei")


def validate_record(record: dict, index: int) -> list[str]:
    errors = []
    prefix = f"records[{index}]"

    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            errors.append(f"{prefix}: missing '{field}'")

    for side in ("l1", "l2"):
        pool = record.get(side, {})
        for field in REQUIRED_POOL_FIELDS:
            if field not in pool:
                errors.append(f"{prefix}.{side}: missing '{field}'")

    gas = record.get("gas", {})
    for field in REQUIRED_GAS_FIELDS:
        if field not in gas:
            errors.append(f"{prefix}.gas: missing '{field}'")

    l1_price = record.get("l1", {}).get("eth_usdc_price")
    l2_price = record.get("l2", {}).get("eth_usdc_price")
    if l1_price and l2_price:
        spread_pct = (l1_price - l2_price) / l2_price * 100
        record_id = record.get("id", "?")
        print(f"  {record_id}: L1={l1_price:.2f}, L2={l2_price:.2f}, spread={spread_pct:+.3f}%")

    return errors


def main():
    if not MOCK_FILE.exists():
        print(f"File not found: {MOCK_FILE}")
        sys.exit(1)

    with open(MOCK_FILE, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    print(f"Loaded {len(records)} records from {MOCK_FILE.name}\n")

    all_errors = []
    ids = set()
    for i, record in enumerate(records):
        rid = record.get("id")
        if rid in ids:
            all_errors.append(f"duplicate id: {rid}")
        ids.add(rid)
        all_errors.extend(validate_record(record, i))

    print()
    if all_errors:
        print("VALIDATION FAILED:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"Validation passed. {len(records)} records OK.")


if __name__ == "__main__":
    main()
