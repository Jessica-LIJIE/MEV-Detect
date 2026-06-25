import json
from pathlib import Path

from src.types import GasState, MarketSnapshot, PoolState

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOCK_PATH = ROOT / "data" / "mock_mempool.json"
DEFAULT_MOCK_MULTIPOOL_PATH = ROOT / "data" / "mock_multipool" / "records.json"


def _parse_pool(data: dict) -> PoolState:
    return PoolState(
        chain=data["chain"],
        block_number=data["block_number"],
        block_time=data["block_time"],
        pool_address=data["pool_address"],
        sqrt_price_x96=int(data["sqrt_price_x96"]),
        liquidity=int(data["liquidity"]),
        tick=data["tick"],
        eth_usdc_price=float(data["eth_usdc_price"]),
    )


def record_to_snapshot(record: dict) -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=record["id"],
        timestamp=record["timestamp"],
        trigger=record["trigger"],
        l1=_parse_pool(record["l1"]),
        l2=_parse_pool(record["l2"]),
        l2_lag_ms=record["l2_lag_ms"],
        gas=GasState(
            l1_gwei=record["gas"]["l1_gwei"],
            l2_gwei=record["gas"]["l2_gwei"],
        ),
        bridge_fee_usd=float(record["bridge_fee_usd"]),
        swap_amount_usd=record.get("swap_amount_usd"),
        scenario=record.get("scenario"),
    )


def load_mock_snapshots(path: Path | None = None) -> list[MarketSnapshot]:
    mock_path = path or DEFAULT_MOCK_PATH
    with open(mock_path, encoding="utf-8") as f:
        data = json.load(f)
    return [record_to_snapshot(r) for r in data["records"]]


def load_mock_multipool_snapshots(path: Path | None = None):
    """加载 Phase 4 Mock 多池快照（与 live RPC 数据分目录）。"""
    from src.multipool_mock import load_mock_multipool_snapshots as _load

    return _load(path or DEFAULT_MOCK_MULTIPOOL_PATH)
