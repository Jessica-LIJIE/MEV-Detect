"""L1/L2 场景化 PSO 参数选择（E3c）。"""

from __future__ import annotations

from config.settings import PSO_CONFIG, PSO_PROFILES, PSO_PROFILE_THRESHOLDS
from src.types import MarketSnapshot


def snapshot_spread_pct(snapshot: MarketSnapshot) -> float:
    l2_price = snapshot.l2.eth_usdc_price
    if not l2_price:
        return 0.0
    return (snapshot.l1.eth_usdc_price - l2_price) / l2_price * 100


def get_fixed_pso_profile() -> dict:
    """固定参数基线，与 PSO_CONFIG 一致。"""
    return {
        "name": "fixed",
        "num_particles": PSO_CONFIG["num_particles"],
        "max_iter": PSO_CONFIG["max_iter"],
        "spread_pct": None,
        "l2_lag_ms": None,
        "reason": "fixed PSO_CONFIG",
    }


def select_pso_profile(snapshot: MarketSnapshot) -> dict:
    """根据价差与 L2 延迟选择 PSO 粒子数与迭代次数。"""
    spread_pct = snapshot_spread_pct(snapshot)
    lag_ms = snapshot.l2_lag_ms
    large_spread_threshold = PSO_PROFILE_THRESHOLDS["large_spread_pct"]
    low_lag_threshold = PSO_PROFILE_THRESHOLDS["low_lag_ms"]

    if spread_pct > large_spread_threshold:
        cfg = PSO_PROFILES["large_spread"]
        return {
            "name": "large_spread",
            "num_particles": cfg["num_particles"],
            "max_iter": cfg["max_iter"],
            "spread_pct": round(spread_pct, 4),
            "l2_lag_ms": lag_ms,
            "reason": f"spread_pct={spread_pct:.3f}% > {large_spread_threshold}%",
        }

    if lag_ms < low_lag_threshold:
        cfg = PSO_PROFILES["low_lag"]
        return {
            "name": "low_lag",
            "num_particles": cfg["num_particles"],
            "max_iter": cfg["max_iter"],
            "spread_pct": round(spread_pct, 4),
            "l2_lag_ms": lag_ms,
            "reason": f"l2_lag_ms={lag_ms} < {low_lag_threshold}ms",
        }

    cfg = PSO_PROFILES["default"]
    return {
        "name": "default",
        "num_particles": cfg["num_particles"],
        "max_iter": cfg["max_iter"],
        "spread_pct": round(spread_pct, 4),
        "l2_lag_ms": lag_ms,
        "reason": "default profile",
    }
