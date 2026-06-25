"""阶段1：跨层多池候选路由筛选。

默认使用 ``stage1_graph`` 完整实现（token 图 + 负对数汇率 + 可采纳松弛）。
设置 ``CYCLE_FINDER_CONFIG['mode']='mvp'`` 可回退笛卡尔积 Top-K。
"""

from __future__ import annotations

from config.settings import BRIDGE_CONFIG, CYCLE_FINDER_CONFIG
from src.types import CandidateRoute, MultiPoolSnapshot, PoolSnapshot

STABLE_QUOTE_TOKENS = frozenset({"USDC", "USDT", "DAI"})


def quote_token(pool: PoolSnapshot) -> str | None:
    """池子中非 WETH 的一腿（用于跨层同资产配对）。"""
    if pool.token0 == "WETH":
        return pool.token1
    if pool.token1 == "WETH":
        return pool.token0
    return None


def is_comparable_pair(l1_pool: PoolSnapshot, l2_pool: PoolSnapshot) -> bool:
    """仅比较同稳定币报价的 WETH/Stable 池（排除 WETH/ARB 等）。"""
    q1, q2 = quote_token(l1_pool), quote_token(l2_pool)
    if q1 is None or q2 is None:
        return False
    if q1 not in STABLE_QUOTE_TOKENS or q2 not in STABLE_QUOTE_TOKENS:
        return False
    return q1 == q2


def liquidity_usd_proxy(
    pool: PoolSnapshot,
    *,
    liquidity_usd_scale: float | None = None,
) -> float:
    """V3 liquidity 粗算 USD 名义深度（供阶段1排序/过滤）。"""
    scale = liquidity_usd_scale or CYCLE_FINDER_CONFIG["liquidity_usd_scale"]
    if scale <= 0 or pool.eth_usdc_price <= 0:
        return 0.0
    return float(pool.liquidity) * pool.eth_usdc_price / scale


def effective_depth_usd(
    pool: PoolSnapshot,
    *,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
) -> float:
    """effective_depth = liquidity × price_impact_threshold（文档 MVP 公式）。"""
    threshold = (
        price_impact_threshold
        if price_impact_threshold is not None
        else CYCLE_FINDER_CONFIG["price_impact_threshold"]
    )
    return liquidity_usd_proxy(pool, liquidity_usd_scale=liquidity_usd_scale) * threshold


def pair_spread_abs(l1_pool: PoolSnapshot, l2_pool: PoolSnapshot) -> float:
    """两池 ETH/USDC 统一报价之绝对价差（USD/ETH）。"""
    return abs(l1_pool.eth_usdc_price - l2_pool.eth_usdc_price)


def static_route_score(
    l1_pool: PoolSnapshot,
    l2_pool: PoolSnapshot,
    *,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
    bridge_fee_usd: float = 0.0,
) -> float:
    """静态分：spread_ij × min(depth_l1, depth_l2) − 微量桥费 tie-break。"""
    spread = pair_spread_abs(l1_pool, l2_pool)
    depth_l1 = effective_depth_usd(
        l1_pool,
        price_impact_threshold=price_impact_threshold,
        liquidity_usd_scale=liquidity_usd_scale,
    )
    depth_l2 = effective_depth_usd(
        l2_pool,
        price_impact_threshold=price_impact_threshold,
        liquidity_usd_scale=liquidity_usd_scale,
    )
    return spread * min(depth_l1, depth_l2) - bridge_fee_usd * 0.01


def _passes_depth_filter(
    pool: PoolSnapshot,
    min_effective_depth_usd: float,
    *,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
) -> bool:
    depth = effective_depth_usd(
        pool,
        price_impact_threshold=price_impact_threshold,
        liquidity_usd_scale=liquidity_usd_scale,
    )
    return depth >= min_effective_depth_usd


def enumerate_scored_routes_mvp(
    snapshot: MultiPoolSnapshot,
    *,
    min_effective_depth_usd: float | None = None,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
    bridges: list[dict] | None = None,
) -> list[tuple[float, int, int, int]]:
    """MVP：枚举通过深度过滤的 (score, l1_idx, l2_idx, bridge_idx)。"""
    min_depth = (
        min_effective_depth_usd
        if min_effective_depth_usd is not None
        else CYCLE_FINDER_CONFIG["min_effective_depth_usd"]
    )
    bridge_list = bridges if bridges is not None else BRIDGE_CONFIG["bridges"]
    scored: list[tuple[float, int, int, int]] = []

    for i, l1_pool in enumerate(snapshot.l1_pools):
        if not _passes_depth_filter(
            l1_pool,
            min_depth,
            price_impact_threshold=price_impact_threshold,
            liquidity_usd_scale=liquidity_usd_scale,
        ):
            continue
        for j, l2_pool in enumerate(snapshot.l2_pools):
            if not is_comparable_pair(l1_pool, l2_pool):
                continue
            if not _passes_depth_filter(
                l2_pool,
                min_depth,
                price_impact_threshold=price_impact_threshold,
                liquidity_usd_scale=liquidity_usd_scale,
            ):
                continue
            for b, bridge in enumerate(bridge_list):
                score = static_route_score(
                    l1_pool,
                    l2_pool,
                    price_impact_threshold=price_impact_threshold,
                    liquidity_usd_scale=liquidity_usd_scale,
                    bridge_fee_usd=float(bridge.get("fee_usd", 0.0)),
                )
                scored.append((score, i, j, b))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def find_top_k_candidates_mvp(
    snapshot: MultiPoolSnapshot,
    *,
    top_k: int | None = None,
    min_effective_depth_usd: float | None = None,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
    bridges: list[dict] | None = None,
) -> list[CandidateRoute]:
    """MVP 笛卡尔积 Top-K。"""
    k = top_k if top_k is not None else CYCLE_FINDER_CONFIG["top_k"]
    bridge_list = bridges if bridges is not None else BRIDGE_CONFIG["bridges"]
    scored = enumerate_scored_routes_mvp(
        snapshot,
        min_effective_depth_usd=min_effective_depth_usd,
        price_impact_threshold=price_impact_threshold,
        liquidity_usd_scale=liquidity_usd_scale,
        bridges=bridge_list,
    )

    routes: list[CandidateRoute] = []
    for route_id, (score, i, j, b) in enumerate(scored[:k]):
        l1_pool = snapshot.l1_pools[i]
        l2_pool = snapshot.l2_pools[j]
        bridge = bridge_list[b]
        routes.append(
            CandidateRoute(
                route_id=route_id,
                l1_pool_idx=i,
                l2_pool_idx=j,
                bridge_idx=b,
                score=score,
                l1_pool_id=l1_pool.pool_id,
                l2_pool_id=l2_pool.pool_id,
                bridge_id=str(bridge.get("id", f"bridge_{b}")),
            )
        )
    return routes


def find_top_k_candidates(
    snapshot: MultiPoolSnapshot,
    *,
    top_k: int | None = None,
    min_effective_depth_usd: float | None = None,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
    bridges: list[dict] | None = None,
    mode: str | None = None,
) -> list[CandidateRoute]:
    """输入 MultiPoolSnapshot，输出 Top-K CandidateRoute（长度 ≤ K）。"""
    finder_mode = mode or CYCLE_FINDER_CONFIG.get("mode", "full")
    if finder_mode == "mvp":
        return find_top_k_candidates_mvp(
            snapshot,
            top_k=top_k,
            min_effective_depth_usd=min_effective_depth_usd,
            price_impact_threshold=price_impact_threshold,
            liquidity_usd_scale=liquidity_usd_scale,
            bridges=bridges,
        )

    from src.stage1_graph import find_graph_candidates

    return find_graph_candidates(
        snapshot,
        top_k=top_k,
        min_effective_depth_usd=min_effective_depth_usd,
        price_impact_threshold=price_impact_threshold,
        liquidity_usd_scale=liquidity_usd_scale,
        bridges=bridges,
    )


# 向后兼容别名
enumerate_scored_routes = enumerate_scored_routes_mvp


def format_candidate_route(route: CandidateRoute, snapshot: MultiPoolSnapshot) -> str:
    """人类可读的单条候选摘要。"""
    l1 = snapshot.l1_pools[route.l1_pool_idx]
    l2 = snapshot.l2_pools[route.l2_pool_idx]
    spread = pair_spread_abs(l1, l2)
    return (
        f"#{route.route_id:02d} score={route.score:,.2f} | "
        f"L1[{route.l1_pool_idx}] {route.l1_pool_id} @ {l1.eth_usdc_price:.2f} | "
        f"L2[{route.l2_pool_idx}] {route.l2_pool_id} @ {l2.eth_usdc_price:.2f} | "
        f"spread={spread:.2f} | bridge={route.bridge_id}"
    )
