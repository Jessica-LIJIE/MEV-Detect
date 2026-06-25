"""Phase 5：Fork / QuoterV2 验真单元与集成测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.fork_verify import (
    AnvilProcess,
    eth_to_wei,
    fork_net_profit_from_gross,
    rel_error,
    stable_to_usd,
    verify_route,
)
from src.models import MultiPoolCostModel
from src.types import CandidateRoute, GasState, InventoryState, MultiPoolSnapshot, PoolSnapshot


def _pool(chain: str, pool_id: str, price: float) -> PoolSnapshot:
    token0, token1 = ("USDC", "WETH") if chain == "ethereum" else ("WETH", "USDC")
    return PoolSnapshot(
        chain=chain,
        pool_address="0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
        if pool_id == "eth_usdc_005"
        else "0xC6962004f452bE9203591991D15f6b388e09E8D0",
        token0=token0,
        token1=token1,
        fee_tier=500,
        sqrt_price_x96=1,
        liquidity=3_000_000_000_000_000_000,
        tick=0,
        eth_usdc_price=price,
        pool_id=pool_id,
        block_number=100,
    )


def _snapshot(l1_price: float = 3600.0, l2_price: float = 3400.0) -> MultiPoolSnapshot:
    return MultiPoolSnapshot(
        snapshot_id="fork_test",
        l1_block=20_100_100,
        l2_block=220_001_500,
        l1_pools=[_pool("ethereum", "eth_usdc_005", l1_price)],
        l2_pools=[_pool("arbitrum", "arb_usdc_005", l2_price)],
        gas=GasState(l1_gwei=15.0, l2_gwei=0.08),
        bridge_fee_usd=5.0,
        inventory=InventoryState(l1_eth=10.0, l2_eth=10.0),
        trigger="test",
        timestamp="2025-06-01T12:00:00Z",
    )


def _route() -> CandidateRoute:
    return CandidateRoute(
        route_id=0,
        l1_pool_idx=0,
        l2_pool_idx=0,
        bridge_idx=1,
        score=1.0,
        l1_pool_id="eth_usdc_005",
        l2_pool_id="arb_usdc_005",
        bridge_id="across",
    )


def test_rel_error_symmetric():
    assert rel_error(100.0, 90.0) == pytest.approx(0.1)
    assert rel_error(-10.0, -9.0) == pytest.approx(0.1)


def test_eth_wei_and_stable_conversion():
    assert eth_to_wei(1.5) == 1_500_000_000_000_000_000
    assert stable_to_usd(1_500_000, 6) == pytest.approx(1.5)


def test_authenticated_fork_url_embeds_infura_secret():
    from src.rpc_utils import authenticated_fork_url

    url = "https://mainnet.infura.io/v3/abc123"
    authed = authenticated_fork_url(url, api_secret="mysecret")
    assert authed.startswith("https://:mysecret@mainnet.infura.io/v3/abc123")
    assert authenticated_fork_url(url, api_secret="") == url
    assert authenticated_fork_url("https://rpc.example.com", api_secret="x") == "https://rpc.example.com"


def test_fork_net_profit_from_gross():
    snapshot = _snapshot()
    route = _route()
    model = MultiPoolCostModel.from_routes(snapshot, [route])
    net = fork_net_profit_from_gross(500.0, snapshot, route, 1.0, model)
    assert net < 500.0


@patch("src.fork_verify.quote_cross_layer_gross_usd")
@patch("src.fork_verify.RPC", {"ethereum": {"http": "http://l1"}, "arbitrum": {"http": "http://l2"}})
@patch("src.fork_verify.create_sync_w3")
def test_verify_route_with_mocked_quoter(mock_w3, mock_gross):
    mock_gross.return_value = (180.0, "mock")
    mock_w3.return_value = MagicMock()

    snapshot = _snapshot()
    route = _route()
    result = verify_route(snapshot, route, 1.0, use_anvil=False)

    assert result.quoter_ok is True
    assert result.fork_profit != 0.0
    assert result.verification_mode == "rpc_quoter"
    assert result.rel_error >= 0.0


@pytest.mark.integration
def test_live_quoter_on_real_snapshot():
    """需 .env RPC；在真实快照上跑 QuoterV2（不要求 anvil）。"""
    from config.settings import RPC
    from pathlib import Path

    if not RPC["ethereum"]["http"] or not RPC["arbitrum"]["http"]:
        pytest.skip("RPC not configured")

    snap_path = Path("data/snapshots/snap_latest.json")
    if not snap_path.is_file():
        pytest.skip("snap_latest.json missing")

    from src.snapshot_builder import load_multi_pool_snapshot
    from src.cycle_finder import find_top_k_candidates

    snapshot = load_multi_pool_snapshot(snap_path)
    routes = find_top_k_candidates(snapshot, top_k=8, min_effective_depth_usd=1.0)
    route = next(
        (r for r in routes if r.l1_pool_id.startswith("eth_usdc") and r.l2_pool_id.startswith("arb_usdc")),
        routes[0],
    )

    result = verify_route(snapshot, route, 0.1, use_anvil=False)
    assert result.quoter_ok, result.notes
    assert result.verification_mode == "rpc_quoter"
    assert result.rel_error >= 0.0


@pytest.mark.integration
def test_live_usdc_route_rel_error_under_10pct():
    """验收：至少一条 USDC 路由 Quoter 与模拟误差 < 10%。"""
    from config.settings import RPC
    from pathlib import Path

    if not RPC["ethereum"]["http"] or not RPC["arbitrum"]["http"]:
        pytest.skip("RPC not configured")

    snap_path = Path("data/snapshots/snap_latest.json")
    if not snap_path.is_file():
        pytest.skip("snap_latest.json missing")

    from src.snapshot_builder import load_multi_pool_snapshot
    from src.cycle_finder import find_top_k_candidates

    snapshot = load_multi_pool_snapshot(snap_path)
    routes = find_top_k_candidates(snapshot, top_k=32, min_effective_depth_usd=1.0)
    route = next(
        (
            r
            for r in routes
            if "usdc" in r.l1_pool_id.lower() and "usdc" in r.l2_pool_id.lower()
        ),
        None,
    )
    if route is None:
        pytest.skip("no USDC comparable route in Top-K")

    result = verify_route(snapshot, route, 0.01, use_anvil=False)
    assert result.quoter_ok, result.notes
    assert result.rel_error < 0.10, (
        f"rel_error={result.rel_error:.2%} sim={result.simulated_profit} fork={result.fork_profit}"
    )
