"""Uniswap V3 池子事件解析与价格换算工具。"""

from dataclasses import dataclass

from eth_abi import decode
from web3 import Web3

from config.settings import POOL_DECIMALS

Q96 = 2**96

SWAP_TOPIC = Web3.keccak(
    text="Swap(address,address,int256,int256,uint160,uint128,int24)"
).to_0x_hex()

POOL_ABI = [
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"name": "sqrtPriceX96", "type": "uint160"},
            {"name": "tick", "type": "int24"},
            {"name": "observationIndex", "type": "uint16"},
            {"name": "observationCardinality", "type": "uint16"},
            {"name": "observationCardinalityNext", "type": "uint16"},
            {"name": "feeProtocol", "type": "uint8"},
            {"name": "unlocked", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "liquidity",
        "outputs": [{"name": "", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "tick", "type": "int24"}],
        "name": "ticks",
        "outputs": [
            {"name": "liquidityGross", "type": "uint128"},
            {"name": "liquidityNet", "type": "int128"},
            {"name": "feeGrowthOutside0X128", "type": "uint256"},
            {"name": "feeGrowthOutside1X128", "type": "uint256"},
            {"name": "tickCumulativeOutside", "type": "int56"},
            {"name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
            {"name": "secondsOutside", "type": "uint32"},
            {"name": "initialized", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

# Uniswap V3 费率档 → tick spacing
TICK_SPACING_BY_FEE = {
    100: 1,
    500: 10,
    3000: 60,
    10000: 200,
}


@dataclass
class SwapEvent:
    amount0: int
    amount1: int
    sqrt_price_x96: int
    liquidity: int
    tick: int
    tx_hash: str
    block_number: int


def sqrt_price_x96_to_eth_usdc(sqrt_price_x96: int, chain: str = "ethereum") -> float:
    """将 sqrtPriceX96 转为 ETH/USDC 人类可读价格（USD per ETH）。"""
    meta = POOL_DECIMALS.get(chain, POOL_DECIMALS["ethereum"])
    return sqrt_price_x96_to_eth_usdc_meta(
        sqrt_price_x96,
        meta["token0"],
        meta["token1"],
        meta.get("weth_is_token0", False),
    )


def sqrt_price_x96_to_eth_usdc_meta(
    sqrt_price_x96: int,
    token0_decimals: int,
    token1_decimals: int,
    weth_is_token0: bool,
) -> float:
    """按池子 token 顺序与小数位换算 ETH/USDC 价格。"""
    if sqrt_price_x96 <= 0:
        return 0.0

    ratio = (sqrt_price_x96 / Q96) ** 2
    if ratio == 0:
        return 0.0

    if weth_is_token0:
        return ratio * (10 ** (token0_decimals - token1_decimals))
    return (1.0 / ratio) * (10 ** (token1_decimals - token0_decimals))


def tick_spacing_for_fee(fee_tier: int) -> int:
    return TICK_SPACING_BY_FEE.get(fee_tier, 60)


def estimate_swap_usd(
    amount0: int,
    amount1: int,
    eth_usdc_price: float,
    chain: str = "ethereum",
) -> float:
    """根据 Swap 事件估算交易额（USD）。"""
    meta = POOL_DECIMALS.get(chain, POOL_DECIMALS["ethereum"])
    dec0, dec1 = meta["token0"], meta["token1"]

    if meta.get("weth_is_token0"):
        eth_amount = abs(amount0) / (10**dec0)
        usdc_amount = abs(amount1) / (10**dec1)
    else:
        usdc_amount = abs(amount0) / (10**dec0)
        eth_amount = abs(amount1) / (10**dec1)

    if usdc_amount >= eth_amount * eth_usdc_price:
        return usdc_amount
    return eth_amount * eth_usdc_price


def decode_swap_log(log: dict) -> SwapEvent:
    """解析 Uniswap V3 Swap 事件日志。"""
    data = log.get("data", "")
    if isinstance(data, str):
        if data.startswith("0x"):
            data = bytes.fromhex(data[2:])
        else:
            data = bytes.fromhex(data)

    amount0, amount1, sqrt_price_x96, liquidity, tick = decode(
        ["int256", "int256", "uint160", "uint128", "int24"],
        data,
    )

    block_number = log.get("blockNumber")
    if hasattr(block_number, "__int__"):
        block_number = int(block_number)

    tx_hash = log.get("transactionHash", "")
    if hasattr(tx_hash, "hex"):
        tx_hash = tx_hash.hex()
    elif isinstance(tx_hash, bytes):
        tx_hash = "0x" + tx_hash.hex()

    return SwapEvent(
        amount0=amount0,
        amount1=amount1,
        sqrt_price_x96=sqrt_price_x96,
        liquidity=liquidity,
        tick=tick,
        tx_hash=str(tx_hash),
        block_number=block_number,
    )
