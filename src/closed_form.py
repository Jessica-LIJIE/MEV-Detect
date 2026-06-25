"""单候选路由闭式/数值上界（E1 质量锚点）。"""

from __future__ import annotations

from src.models import MultiPoolCostModel
from src.types import CandidateRoute, MultiPoolSnapshot


def golden_section_maximize(
    f,
    low: float,
    high: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 80,
) -> tuple[float, float]:
    """在 [low, high] 上最大化一元函数 f，返回 (best_x, best_value)。"""
    if high < low:
        low, high = high, low
    if high - low < tol:
        mid = (low + high) / 2
        return mid, f(mid)

    gr = 0.618033988749895
    a, b = low, high
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)

    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)

    best_x = (a + b) / 2
    return best_x, f(best_x)


def optimize_route_amount(
    route: CandidateRoute,
    snapshot: MultiPoolSnapshot,
    model: MultiPoolCostModel | None = None,
    *,
    amount_bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """对单条候选路由做黄金分割，返回 (best_amount_eth, best_net_profit_usd)。"""
    model = model or MultiPoolCostModel.from_routes(snapshot, [route])
    low, high = amount_bounds or model.amount_bounds_tuple(snapshot)
    if high <= low:
        return low, model.evaluate_route_net_scalar(low, route, snapshot)

    def objective(amount: float) -> float:
        return model.evaluate_route_net_scalar(amount, route, snapshot)

    return golden_section_maximize(objective, low, high)


def closed_form_upper_bound(
    route: CandidateRoute,
    snapshot: MultiPoolSnapshot,
    model: MultiPoolCostModel | None = None,
) -> dict[str, float]:
    """E1 锚点：单路由最优 amount 与净利润。"""
    amount, profit = optimize_route_amount(route, snapshot, model=model)
    return {
        "amount_in_eth": amount,
        "net_profit_usd": profit,
        "route_id": float(route.route_id),
    }


def global_closed_form_optimum(
    routes: list[CandidateRoute],
    snapshot: MultiPoolSnapshot,
    model: MultiPoolCostModel | None = None,
    *,
    latency_risk_lambda: float | None = None,
) -> tuple[float, float, CandidateRoute | None]:
    """对所有候选路由做闭式优化，返回 (best_profit, best_amount, best_route)。"""
    if not routes:
        return float("-inf"), 0.0, None

    model = model or MultiPoolCostModel.from_routes(
        snapshot,
        routes,
        latency_risk_lambda=latency_risk_lambda,
    )
    best_profit = float("-inf")
    best_amount = 0.0
    best_route: CandidateRoute | None = None
    for route in routes:
        amount, profit = optimize_route_amount(route, snapshot, model)
        if profit > best_profit:
            best_profit = profit
            best_amount = amount
            best_route = route
    return best_profit, best_amount, best_route
