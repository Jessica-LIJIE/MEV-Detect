"""阶段1 完整实现：跨层 token-池图、负对数汇率与可采纳松弛路径检测。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config.settings import BRIDGE_CONFIG, CYCLE_FINDER_CONFIG
from src.types import CandidateRoute, MultiPoolSnapshot, PoolSnapshot

L1_CHAIN = "ethereum"
L2_CHAIN = "arbitrum"
BASE_TOKEN = "WETH"
STABLE_QUOTE_TOKENS = frozenset({"USDC", "USDT", "DAI"})


@dataclass(frozen=True, slots=True)
class TokenNode:
    chain: str
    token: str


@dataclass(slots=True)
class GraphEdge:
    """有向边；weight = -log(effective_exchange_rate)。"""

    src: TokenNode
    dst: TokenNode
    weight: float
    edge_type: str  # pool | bridge
    pool_idx: int | None = None
    pool_chain: str | None = None
    bridge_idx: int | None = None


@dataclass(slots=True)
class CrossLayerPath:
    """跨层 WETH→…→WETH 路径（可含同链多跳）。"""

    edges: list[GraphEdge] = field(default_factory=list)
    log_gain: float = 0.0
    l1_pool_idx: int = -1
    l2_pool_idx: int = -1
    bridge_idx: int = -1
    direction: str = "l1_to_l2"  # l1_to_l2 | l2_to_l1

    @property
    def route_key(self) -> tuple[int, int, int]:
        return self.l1_pool_idx, self.l2_pool_idx, self.bridge_idx


def quote_token(pool: PoolSnapshot) -> str | None:
    if pool.token0 == BASE_TOKEN:
        return pool.token1
    if pool.token1 == BASE_TOKEN:
        return pool.token0
    return None


def is_comparable_pair(l1_pool: PoolSnapshot, l2_pool: PoolSnapshot) -> bool:
    q1, q2 = quote_token(l1_pool), quote_token(l2_pool)
    if q1 is None or q2 is None:
        return False
    if q1 not in STABLE_QUOTE_TOKENS or q2 not in STABLE_QUOTE_TOKENS:
        return False
    return q1 == q2


def pool_swap_rate(pool: PoolSnapshot, token_in: str, token_out: str) -> float:
    """单位输入 token 的输出数量（含费率，不含滑点）。"""
    if pool.eth_usdc_price <= 0:
        return 0.0
    fee = pool.fee_tier / 1_000_000
    factor = 1.0 - fee
    if token_in == BASE_TOKEN and token_out in STABLE_QUOTE_TOKENS:
        return pool.eth_usdc_price * factor
    if token_in in STABLE_QUOTE_TOKENS and token_out == BASE_TOKEN:
        return (1.0 / pool.eth_usdc_price) * factor
    return 0.0


def negative_log_rate(rate: float) -> float:
    if rate <= 0.0:
        return float("inf")
    return -math.log(rate)


def bridge_exchange_rate(fee_usd: float, *, ref_notional_usd: float) -> float:
    """桥接有效传递率（参考名义金额上的乘性损耗）。"""
    if ref_notional_usd <= 0:
        return 1.0
    return max(1.0 - fee_usd / ref_notional_usd, 1e-12)


def liquidity_usd_proxy(
    pool: PoolSnapshot,
    *,
    liquidity_usd_scale: float | None = None,
) -> float:
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
    threshold = (
        price_impact_threshold
        if price_impact_threshold is not None
        else CYCLE_FINDER_CONFIG["price_impact_threshold"]
    )
    return liquidity_usd_proxy(pool, liquidity_usd_scale=liquidity_usd_scale) * threshold


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


def build_token_graph(
    snapshot: MultiPoolSnapshot,
    *,
    bridges: list[dict] | None = None,
    ref_notional_usd: float | None = None,
) -> dict[TokenNode, list[GraphEdge]]:
    """构建跨层 token-池图：池子双向边 + 同资产桥边。"""
    bridge_list = bridges if bridges is not None else BRIDGE_CONFIG["bridges"]
    ref_usd = ref_notional_usd or CYCLE_FINDER_CONFIG.get("bridge_ref_notional_usd", 10_000.0)
    adjacency: dict[TokenNode, list[GraphEdge]] = {}

    def add_edge(edge: GraphEdge) -> None:
        adjacency.setdefault(edge.src, []).append(edge)

    for idx, pool in enumerate(snapshot.l1_pools):
        quote = quote_token(pool)
        if quote is None or quote not in STABLE_QUOTE_TOKENS:
            continue
        for token_in, token_out in ((BASE_TOKEN, quote), (quote, BASE_TOKEN)):
            rate = pool_swap_rate(pool, token_in, token_out)
            if rate <= 0:
                continue
            add_edge(
                GraphEdge(
                    src=TokenNode(L1_CHAIN, token_in),
                    dst=TokenNode(L1_CHAIN, token_out),
                    weight=negative_log_rate(rate),
                    edge_type="pool",
                    pool_idx=idx,
                    pool_chain=L1_CHAIN,
                )
            )

    for idx, pool in enumerate(snapshot.l2_pools):
        quote = quote_token(pool)
        if quote is None or quote not in STABLE_QUOTE_TOKENS:
            continue
        for token_in, token_out in ((BASE_TOKEN, quote), (quote, BASE_TOKEN)):
            rate = pool_swap_rate(pool, token_in, token_out)
            if rate <= 0:
                continue
            add_edge(
                GraphEdge(
                    src=TokenNode(L2_CHAIN, token_in),
                    dst=TokenNode(L2_CHAIN, token_out),
                    weight=negative_log_rate(rate),
                    edge_type="pool",
                    pool_idx=idx,
                    pool_chain=L2_CHAIN,
                )
            )

    bridge_tokens = set(STABLE_QUOTE_TOKENS) | {BASE_TOKEN}
    for b_idx, bridge in enumerate(bridge_list):
        fee_usd = float(bridge.get("fee_usd", 0.0))
        rate = bridge_exchange_rate(fee_usd, ref_notional_usd=ref_usd)
        weight = negative_log_rate(rate)
        for token in bridge_tokens:
            add_edge(
                GraphEdge(
                    src=TokenNode(L1_CHAIN, token),
                    dst=TokenNode(L2_CHAIN, token),
                    weight=weight,
                    edge_type="bridge",
                    bridge_idx=b_idx,
                )
            )
            add_edge(
                GraphEdge(
                    src=TokenNode(L2_CHAIN, token),
                    dst=TokenNode(L1_CHAIN, token),
                    weight=weight,
                    edge_type="bridge",
                    bridge_idx=b_idx,
                )
            )

    return adjacency


def _extract_route_indices(path: CrossLayerPath) -> CrossLayerPath | None:
    """从路径边序列解析 (l1_pool, l2_pool, bridge) 三元组。"""
    l1_pool: int | None = None
    l2_pool: int | None = None
    bridge_idx: int | None = None
    direction = path.direction

    for edge in path.edges:
        if edge.edge_type == "bridge" and bridge_idx is None:
            bridge_idx = edge.bridge_idx
        elif edge.edge_type == "pool" and edge.pool_chain == L1_CHAIN:
            l1_pool = edge.pool_idx
        elif edge.edge_type == "pool" and edge.pool_chain == L2_CHAIN:
            l2_pool = edge.pool_idx

    if l1_pool is None or l2_pool is None or bridge_idx is None:
        return None

    path.l1_pool_idx = l1_pool
    path.l2_pool_idx = l2_pool
    path.bridge_idx = bridge_idx
    path.direction = direction
    return path


def enumerate_cross_layer_paths(
    adjacency: dict[TokenNode, list[GraphEdge]],
    *,
    max_hops: int | None = None,
) -> list[CrossLayerPath]:
    """BFS 枚举 L1 WETH ↔ L2 WETH 跨层路径（含同链多跳）。"""
    hop_limit = max_hops or CYCLE_FINDER_CONFIG.get("max_path_hops", 6)
    results: list[CrossLayerPath] = []

    searches = (
        (TokenNode(L1_CHAIN, BASE_TOKEN), TokenNode(L2_CHAIN, BASE_TOKEN), "l1_to_l2"),
        (TokenNode(L2_CHAIN, BASE_TOKEN), TokenNode(L1_CHAIN, BASE_TOKEN), "l2_to_l1"),
    )

    for start, goal, direction in searches:
        queue: list[tuple[TokenNode, list[GraphEdge], bool]] = [(start, [], False)]
        while queue:
            node, edges, used_bridge = queue.pop(0)
            if len(edges) > hop_limit:
                continue
            if node == goal and used_bridge and edges:
                total_weight = sum(e.weight for e in edges)
                candidate = CrossLayerPath(
                    edges=list(edges),
                    log_gain=-total_weight,
                    direction=direction,
                )
                parsed = _extract_route_indices(candidate)
                if parsed is not None:
                    results.append(parsed)
                continue

            for edge in adjacency.get(node, []):
                next_bridge = used_bridge or edge.edge_type == "bridge"
                if len(edges) + 1 > hop_limit:
                    continue
                queue.append((edge.dst, edges + [edge], next_bridge))

    return results


def path_static_score(
    path: CrossLayerPath,
    snapshot: MultiPoolSnapshot,
    *,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
    bridges: list[dict] | None = None,
) -> float:
    """负对数增益 × 有效深度 − 桥费 tie-break（与 MVP score 单调一致）。"""
    bridge_list = bridges if bridges is not None else BRIDGE_CONFIG["bridges"]
    l1 = snapshot.l1_pools[path.l1_pool_idx]
    l2 = snapshot.l2_pools[path.l2_pool_idx]
    depth_l1 = effective_depth_usd(
        l1,
        price_impact_threshold=price_impact_threshold,
        liquidity_usd_scale=liquidity_usd_scale,
    )
    depth_l2 = effective_depth_usd(
        l2,
        price_impact_threshold=price_impact_threshold,
        liquidity_usd_scale=liquidity_usd_scale,
    )
    bridge_fee = 0.0
    if 0 <= path.bridge_idx < len(bridge_list):
        bridge_fee = float(bridge_list[path.bridge_idx].get("fee_usd", 0.0))
    return path.log_gain * min(depth_l1, depth_l2) - bridge_fee * 0.01


def find_graph_candidates(
    snapshot: MultiPoolSnapshot,
    *,
    top_k: int | None = None,
    min_effective_depth_usd: float | None = None,
    price_impact_threshold: float | None = None,
    liquidity_usd_scale: float | None = None,
    bridges: list[dict] | None = None,
    admissible_relaxation_eps: float | None = None,
    depth_relaxation: float | None = None,
    max_hops: int | None = None,
) -> list[CandidateRoute]:
    """负对数图 + 可采纳松弛 → Top-K CandidateRoute。"""
    k = top_k if top_k is not None else CYCLE_FINDER_CONFIG["top_k"]
    min_depth = (
        min_effective_depth_usd
        if min_effective_depth_usd is not None
        else CYCLE_FINDER_CONFIG["min_effective_depth_usd"]
    )
    depth_relax = (
        depth_relaxation
        if depth_relaxation is not None
        else CYCLE_FINDER_CONFIG.get("depth_relaxation", 0.15)
    )
    relax_eps = (
        admissible_relaxation_eps
        if admissible_relaxation_eps is not None
        else CYCLE_FINDER_CONFIG.get("admissible_relaxation_eps", 0.002)
    )
    relaxed_min_depth = min_depth * max(1.0 - depth_relax, 0.0)

    adjacency = build_token_graph(snapshot, bridges=bridges)
    raw_paths = enumerate_cross_layer_paths(adjacency, max_hops=max_hops)

    best_by_key: dict[tuple[int, int, int], CrossLayerPath] = {}
    for path in raw_paths:
        l1 = snapshot.l1_pools[path.l1_pool_idx]
        l2 = snapshot.l2_pools[path.l2_pool_idx]
        if not is_comparable_pair(l1, l2):
            continue
        if path.log_gain < -relax_eps:
            continue
        if not _passes_depth_filter(
            l1,
            relaxed_min_depth,
            price_impact_threshold=price_impact_threshold,
            liquidity_usd_scale=liquidity_usd_scale,
        ):
            continue
        if not _passes_depth_filter(
            l2,
            relaxed_min_depth,
            price_impact_threshold=price_impact_threshold,
            liquidity_usd_scale=liquidity_usd_scale,
        ):
            continue

        key = path.route_key
        prev = best_by_key.get(key)
        if prev is None or path.log_gain > prev.log_gain:
            best_by_key[key] = path

    bridge_list = bridges if bridges is not None else BRIDGE_CONFIG["bridges"]
    scored: list[tuple[float, CrossLayerPath]] = []
    for path in best_by_key.values():
        score = path_static_score(
            path,
            snapshot,
            price_impact_threshold=price_impact_threshold,
            liquidity_usd_scale=liquidity_usd_scale,
            bridges=bridge_list,
        )
        scored.append((score, path))

    scored.sort(key=lambda item: item[0], reverse=True)

    routes: list[CandidateRoute] = []
    for route_id, (score, path) in enumerate(scored[:k]):
        l1_pool = snapshot.l1_pools[path.l1_pool_idx]
        l2_pool = snapshot.l2_pools[path.l2_pool_idx]
        bridge = bridge_list[path.bridge_idx]
        routes.append(
            CandidateRoute(
                route_id=route_id,
                l1_pool_idx=path.l1_pool_idx,
                l2_pool_idx=path.l2_pool_idx,
                bridge_idx=path.bridge_idx,
                score=score,
                l1_pool_id=l1_pool.pool_id,
                l2_pool_id=l2_pool.pool_id,
                bridge_id=str(bridge.get("id", f"bridge_{path.bridge_idx}")),
            )
        )
    return routes
