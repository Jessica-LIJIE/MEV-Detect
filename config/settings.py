import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
POOL_REGISTRY_PATH = ROOT / "data" / "pool_registry.json"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
MOCK_MULTIPOOL_DIR = ROOT / "data" / "mock_multipool"

# Infura 若开启「Require API Key Secret」，需在 .env 填写 INFURA_API_KEY_SECRET
INFURA_API_KEY_SECRET = os.getenv("INFURA_API_KEY_SECRET", "")

RPC = {
    "ethereum": {
        "http": os.getenv("ETH_HTTP_URL", ""),
        "ws": os.getenv("ETH_WS_URL", ""),
    },
    "arbitrum": {
        "http": os.getenv("ARB_HTTP_URL", ""),
        "ws": os.getenv("ARB_WS_URL", ""),
    },
}

POOLS = {
    "ethereum": {
        # Uniswap V3 USDC/WETH 0.05% — token0=USDC, token1=WETH
        "eth_usdc_005": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    },
    "arbitrum": {
        # Uniswap V3 WETH/USDC 0.05% — token0=WETH, token1=USDC
        "eth_usdc_005": "0xC6962004f452bE9203591991D15f6b388e09E8D0",
    },
}

# token0/token1 小数位，用于价格换算（各链池子 token 顺序不同）
POOL_DECIMALS = {
    "ethereum": {"token0": 6, "token1": 18, "weth_is_token0": False},
    "arbitrum": {"token0": 18, "token1": 6, "weth_is_token0": True},
}

PSO_CONFIG = {
    "num_particles": 1000,
    "max_iter": 100,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5,
    "device": os.getenv("DEVICE", "cpu"),
}

# E3c：自适应 PSO 阈值与档位（default 与 PSO_CONFIG 保持一致）
PSO_PROFILE_THRESHOLDS = {
    "large_spread_pct": 1.0,
    "low_lag_ms": 500,
}

PSO_PROFILES = {
    "large_spread": {"num_particles": 1500, "max_iter": 120},
    "low_lag": {"num_particles": 500, "max_iter": 60},
    "default": {"num_particles": PSO_CONFIG["num_particles"], "max_iter": PSO_CONFIG["max_iter"]},
}

LISTENER_CONFIG = {
    "l1_swap_threshold_usd": 50000,
    "reconnect_delay_sec": 5,
    "max_reconnect_attempts": 10,
}

COST_DEFAULTS = {
    "l1_gas_limit": 200000,
    "l2_gas_limit": 500000,
    "bridge_fee_usd": 5.0,
}

# Phase 0+：跨层桥参数（MVP 使用固定配置，Phase 5 可对齐链上合约）
BRIDGE_CONFIG = {
    "bridges": [
        {
            "id": "canonical",
            "name": "Arbitrum canonical bridge (slow withdrawal)",
            "fee_usd": 2.0,
            "latency_hours": 168.0,
        },
        {
            "id": "across",
            "name": "Across fast bridge",
            "fee_usd": 8.0,
            "latency_hours": 0.05,
        },
        {
            "id": "hop",
            "name": "Hop fast bridge",
            "fee_usd": 6.0,
            "latency_hours": 0.1,
        },
    ],
    "latency_risk_lambda": 0.01,
}

# 分层库存默认上限（ETH），Phase 3 fitness 使用
INVENTORY_DEFAULT = {
    "l1_eth": 50.0,
    "l2_eth": 50.0,
}

# 阶段1 候选环筛选（默认：负对数 token 图；可选 MVP 笛卡尔积）
CYCLE_FINDER_CONFIG = {
    "mode": "full",  # full | mvp
    "top_k": 32,
    "min_effective_depth_usd": 50_000.0,
    # effective_depth_usd = (liquidity / liquidity_scale) * price * threshold
    "price_impact_threshold": 0.01,
    "liquidity_usd_scale": 1e12,
    # 完整阶段1：负对数图 + 可采纳松弛
    "max_path_hops": 6,
    "admissible_relaxation_eps": 0.002,
    "depth_relaxation": 0.15,
    "bridge_ref_notional_usd": 10_000.0,
}
