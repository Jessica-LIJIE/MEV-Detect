import sys
from pathlib import Path

import pytest
from web3 import Web3

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import RPC
from src.listener import MockListener, MultiChainListener
from src.pool_utils import (
    SWAP_TOPIC,
    decode_swap_log,
    estimate_swap_usd,
    sqrt_price_x96_to_eth_usdc,
)


def _make_swap_log(
    amount0: int,
    amount1: int,
    sqrt_price_x96: int,
    liquidity: int = 10**18,
    tick: int = 201000,
    block_number: int = 20100000,
) -> dict:
  from eth_abi import encode

  data = encode(
      ["int256", "int256", "uint160", "uint128", "int24"],
      [amount0, amount1, sqrt_price_x96, liquidity, tick],
  )
  return {
      "data": "0x" + data.hex(),
      "blockNumber": block_number,
      "transactionHash": "0x" + "ab" * 32,
      "topics": [
          SWAP_TOPIC,
          "0x" + "00" * 32,
          "0x" + "00" * 32,
      ],
  }


def test_sqrt_price_conversion():
    # ETH≈3500 USDC 时，sqrtPriceX96 ≈ sqrt(1e18 / 3500e6) * 2^96
    import math

    q96 = 2**96
    price_raw = 1e18 / (3500 * 1e6)
    sqrt_p = int(math.sqrt(price_raw) * q96)
    price = sqrt_price_x96_to_eth_usdc(sqrt_p)
    assert 3400 < price < 3600


def test_estimate_swap_usd_usdc_side():
    # 100,000 USDC in (6 decimals)
    usd = estimate_swap_usd(-100_000_000_000, 0, 3500.0)
    assert usd == pytest.approx(100_000.0)


def test_decode_swap_log():
    sqrt_p = 79000000000000000000000000000
    log = _make_swap_log(
        amount0=-50_000_000_000,
        amount1=14_000_000_000_000_000_000,
        sqrt_price_x96=sqrt_p,
    )
    swap = decode_swap_log(log)
    assert swap.sqrt_price_x96 == sqrt_p
    assert swap.block_number == 20100000


@pytest.mark.asyncio
async def test_mock_listener_replay():
    received = []

    async def on_snapshot(snapshot):
        received.append(snapshot.snapshot_id)

    listener = MockListener()
    listener.on_snapshot(on_snapshot)
    count = await listener.replay(record_id="snap_001", interval_sec=0)
    assert count == 1
    assert received == ["snap_001"]


def test_multi_chain_listener_not_configured_by_default():
    listener = MultiChainListener()
    if RPC["ethereum"]["ws"] and RPC["arbitrum"]["ws"]:
        assert listener.is_configured()
    else:
        assert not listener.is_configured()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not RPC["ethereum"]["http"] or not RPC["arbitrum"]["http"],
    reason="需要配置 ETH_HTTP_URL 与 ARB_HTTP_URL",
)
async def test_rpc_fetch_latest_snapshot():
    listener = MultiChainListener()
    snapshot = await listener.get_latest_snapshot()
    assert snapshot.l1.eth_usdc_price > 0
    assert snapshot.l2.eth_usdc_price > 0
    assert snapshot.l1.block_number > 0
    assert snapshot.l2.block_number > 0
