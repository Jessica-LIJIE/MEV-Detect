"""V3 简化 swap 模拟（常数乘积 + 费率 + 流动性滑点，MVP）。"""

from __future__ import annotations

from typing import Any

from web3 import Web3

from src.pool_registry import get_contract_addresses
from src.types import PoolSnapshot

FEE_DIVISOR = 1_000_000
DEFAULT_LIQUIDITY_SCALE = 1e15

QUOTER_V2_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
                "name": "params",
                "type": "tuple",
            }
        ],
        "name": "quoteExactInputSingle",
        "outputs": [
            {"name": "amountOut", "type": "uint256"},
            {"name": "sqrtPriceX96After", "type": "uint160"},
            {"name": "initializedTicksCrossed", "type": "uint32"},
            {"name": "gasEstimate", "type": "uint256"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


def pool_fee_rate(pool: PoolSnapshot) -> float:
    return pool.fee_tier / FEE_DIVISOR


def slippage_factor(
    amount_eth: float,
    liquidity: int,
    *,
    liquidity_scale: float = DEFAULT_LIQUIDITY_SCALE,
) -> float:
    """流动性越深，滑点越小（MVP 启发式）。"""
    if amount_eth <= 0:
        return 1.0
    liq_proxy = max(float(liquidity) / liquidity_scale, 1.0)
    return 1.0 / (1.0 + amount_eth / liq_proxy)


def estimate_sell_eth_for_stable_usd(
    amount_eth: float,
    pool: PoolSnapshot,
    *,
    liquidity_scale: float = DEFAULT_LIQUIDITY_SCALE,
) -> float:
    """在池子卖出 ETH，得到稳定币（USD 名义）。"""
    if amount_eth <= 0 or pool.eth_usdc_price <= 0:
        return 0.0
    fee = pool_fee_rate(pool)
    slip = slippage_factor(amount_eth, pool.liquidity, liquidity_scale=liquidity_scale)
    return amount_eth * pool.eth_usdc_price * (1.0 - fee) * slip


def estimate_buy_eth_cost_usd(
    amount_eth: float,
    pool: PoolSnapshot,
    *,
    liquidity_scale: float = DEFAULT_LIQUIDITY_SCALE,
) -> float:
    """在池子买入 ETH，稳定币成本（USD 名义）。"""
    if amount_eth <= 0 or pool.eth_usdc_price <= 0:
        return 0.0
    fee = pool_fee_rate(pool)
    slip = slippage_factor(amount_eth, pool.liquidity, liquidity_scale=liquidity_scale)
    denom = max((1.0 - fee) * slip, 1e-9)
    return amount_eth * pool.eth_usdc_price / denom


def estimate_cross_layer_gross_usd(
    amount_eth: float,
    l1_pool: PoolSnapshot,
    l2_pool: PoolSnapshot,
    *,
    liquidity_scale: float = DEFAULT_LIQUIDITY_SCALE,
) -> float:
    """跨层两腿毛利润：在低价链买 ETH、高价链卖 ETH。"""
    if amount_eth <= 0:
        return 0.0
    spread = l1_pool.eth_usdc_price - l2_pool.eth_usdc_price
    if spread > 0:
        cost = estimate_buy_eth_cost_usd(
            amount_eth, l2_pool, liquidity_scale=liquidity_scale
        )
        revenue = estimate_sell_eth_for_stable_usd(
            amount_eth, l1_pool, liquidity_scale=liquidity_scale
        )
        return revenue - cost
    if spread < 0:
        cost = estimate_buy_eth_cost_usd(
            amount_eth, l1_pool, liquidity_scale=liquidity_scale
        )
        revenue = estimate_sell_eth_for_stable_usd(
            amount_eth, l2_pool, liquidity_scale=liquidity_scale
        )
        return revenue - cost
    return 0.0


def estimate_amount_out(
    amount_in: float,
    pool: PoolSnapshot,
    *,
    token_in: str = "WETH",
    liquidity_scale: float = DEFAULT_LIQUIDITY_SCALE,
) -> float:
    """统一接口：WETH→稳定币 或 稳定币(USD)→WETH。"""
    if amount_in <= 0:
        return 0.0
    if token_in == "WETH":
        return estimate_sell_eth_for_stable_usd(
            amount_in, pool, liquidity_scale=liquidity_scale
        )
    eth_amount = amount_in / max(pool.eth_usdc_price, 1e-9)
    cost = estimate_buy_eth_cost_usd(
        eth_amount, pool, liquidity_scale=liquidity_scale
    )
    if cost <= 0:
        return 0.0
    return eth_amount


def quote_via_quoter_v2(
    w3: Any,
    chain: str,
    token_in: str,
    token_out: str,
    amount_in_wei: int,
    fee_tier: int,
    *,
    block_identifier: int | None = None,
) -> int | None:
    """可选：QuoterV2 staticcall；失败或无 RPC 时返回 None。"""
    try:
        addrs = get_contract_addresses(chain)
        quoter = addrs.get("quoter_v2")
        if not quoter or not w3:
            return None
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(quoter),
            abi=QUOTER_V2_ABI,
        )
        params = (
            Web3.to_checksum_address(token_in),
            Web3.to_checksum_address(token_out),
            int(amount_in_wei),
            int(fee_tier),
            0,
        )
        call_kwargs: dict[str, Any] = {}
        if block_identifier is not None:
            call_kwargs["block_identifier"] = block_identifier
        amount_out, _, _, _ = contract.functions.quoteExactInputSingle(params).call(
            **call_kwargs
        )
        return int(amount_out)
    except Exception:
        return None
