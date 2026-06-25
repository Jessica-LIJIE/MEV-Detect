import torch

from config.settings import COST_DEFAULTS
from src.types import GasState, MarketSnapshot

NUM_ROUTES_L1 = 3
NUM_ROUTES_L2 = 3
NUM_BRIDGES = 3

ROUTE_L2_EFFICIENCY = (1.0, 0.92, 0.85)
ROUTE_L1_GAS_MULT = (1.0, 1.08, 1.15)
BRIDGE_FEE_MULT = (1.0, 1.5, 0.6)

class CostModel:
    def __init__(self, config: dict | None = None):
        cfg = config or COST_DEFAULTS
        self.l1_gas_limit = cfg["l1_gas_limit"]
        self.l2_gas_limit = cfg["l2_gas_limit"]
        self.default_bridge_fee = cfg["bridge_fee_usd"]

    def estimate_gas_l1(self, gas: GasState, eth_price: float, route_l1: int = 0) -> float:
        mult = ROUTE_L1_GAS_MULT[route_l1 % NUM_ROUTES_L1]
        return self.l1_gas_limit * gas.l1_gwei * 1e-9 * eth_price * mult

    def estimate_gas_l2(self, gas: GasState, eth_price: float) -> float:
        return self.l2_gas_limit * gas.l2_gwei * 1e-9 * eth_price

    def estimate_bridge_fee(self, bridge_path: int, base_fee: float) -> float:
        return base_fee * BRIDGE_FEE_MULT[bridge_path % NUM_BRIDGES]

    def calc_gross_profit(
        self,
        amount_in_eth: float,
        l1_price: float,
        l2_price: float,
        route_l2: int,
    ) -> float:
        spread = l1_price - l2_price
        if spread <= 0:
            return 0.0
        efficiency = ROUTE_L2_EFFICIENCY[route_l2 % NUM_ROUTES_L2]
        slippage = 1.0 / (1.0 + amount_in_eth * 0.002)
        return amount_in_eth * spread * efficiency * slippage

    def evaluate_fitness_scalar(
        self,
        amount_in: float,
        route_l1: int,
        route_l2: int,
        bridge_path: int,
        snapshot: MarketSnapshot,
    ) -> float:
        gross = self.calc_gross_profit(
            amount_in,
            snapshot.l1.eth_usdc_price,
            snapshot.l2.eth_usdc_price,
            route_l2,
        )
        gas_l1 = self.estimate_gas_l1(snapshot.gas, snapshot.l1.eth_usdc_price, route_l1)
        gas_l2 = self.estimate_gas_l2(snapshot.gas, snapshot.l2.eth_usdc_price)
        bridge = self.estimate_bridge_fee(bridge_path, snapshot.bridge_fee_usd)
        return gross - gas_l1 - gas_l2 - bridge

    def evaluate_fitness(
        self,
        positions: torch.Tensor,
        snapshot: MarketSnapshot,
    ) -> torch.Tensor:
        """向量化适应度：Fitness = Profit - Gas_L1 - Gas_L2 - Bridge_Fee"""
        device = positions.device
        amount_in = positions[:, 0]
        route_l1 = positions[:, 1].floor().long().clamp(0, NUM_ROUTES_L1 - 1)
        route_l2 = positions[:, 2].floor().long().clamp(0, NUM_ROUTES_L2 - 1)
        bridge = positions[:, 3].floor().long().clamp(0, NUM_BRIDGES - 1)

        l1_price = snapshot.l1.eth_usdc_price
        l2_price = snapshot.l2.eth_usdc_price
        spread = l1_price - l2_price

        route_l2_eff = torch.tensor(ROUTE_L2_EFFICIENCY, device=device, dtype=positions.dtype)[
            route_l2
        ]
        slippage = 1.0 / (1.0 + amount_in * 0.002)
        gross = amount_in * spread * route_l2_eff * slippage
        gross = torch.clamp(gross, min=0.0)

        route_l1_mult = torch.tensor(ROUTE_L1_GAS_MULT, device=device, dtype=positions.dtype)[
            route_l1
        ]
        gas_l1 = (
            self.l1_gas_limit
            * snapshot.gas.l1_gwei
            * 1e-9
            * l1_price
            * route_l1_mult
        )
        gas_l2 = self.l2_gas_limit * snapshot.gas.l2_gwei * 1e-9 * l2_price

        bridge_mult = torch.tensor(BRIDGE_FEE_MULT, device=device, dtype=positions.dtype)[bridge]
        bridge_fee = snapshot.bridge_fee_usd * bridge_mult

        return gross - gas_l1 - gas_l2 - bridge_fee


def get_search_bounds(device: str = "cpu") -> torch.Tensor:
    """shape [4, 2]: amount_in, route_l1, route_l2, bridge_path"""
    bounds = torch.tensor(
        [
            [0.1, 50.0],
            [0.0, NUM_ROUTES_L1 - 0.01],
            [0.0, NUM_ROUTES_L2 - 0.01],
            [0.0, NUM_BRIDGES - 0.01],
        ],
        dtype=torch.float32,
    )
    return bounds.to(device)


def build_strategy(snapshot: MarketSnapshot, result) -> "ArbitrageStrategy":
    from src.types import ArbitrageStrategy

    pos = result.best_position
    amount_in = float(pos[0])
    route_l1 = int(float(pos[1]))
    route_l2 = int(float(pos[2]))
    bridge = int(float(pos[3]))

    model = CostModel()
    gas_l1 = model.estimate_gas_l1(snapshot.gas, snapshot.l1.eth_usdc_price, route_l1)
    gas_l2 = model.estimate_gas_l2(snapshot.gas, snapshot.l2.eth_usdc_price)
    bridge_fee = model.estimate_bridge_fee(bridge, snapshot.bridge_fee_usd)

    return ArbitrageStrategy(
        snapshot_id=snapshot.snapshot_id,
        amount_in_eth=amount_in,
        route_l1=route_l1,
        route_l2=route_l2,
        bridge_path=bridge,
        expected_profit_usd=result.best_fitness,
        gas_cost_usd=gas_l1 + gas_l2,
        bridge_fee_usd=bridge_fee,
        search_elapsed_ms=result.elapsed_ms,
        timestamp=snapshot.timestamp,
    )


class MultiPoolCostModel:
    """阶段2：多池快照 + Top-K 候选的批量 fitness（[N,K] 利润矩阵）。"""

    def __init__(
        self,
        routes: list["CandidateRoute"],
        config: dict | None = None,
        bridges: list[dict] | None = None,
        *,
        latency_risk_lambda: float | None = None,
        volatility_proxy: float = 0.02,
        liquidity_scale: float = 1e15,
    ):
        from src.types import CandidateRoute

        cfg = config or COST_DEFAULTS
        self.l1_gas_limit = cfg["l1_gas_limit"]
        self.l2_gas_limit = cfg["l2_gas_limit"]
        self.default_bridge_fee = cfg["bridge_fee_usd"]
        self.routes = routes
        self.num_candidates = len(routes)
        self.bridges = bridges or []
        self.latency_risk_lambda = latency_risk_lambda or 0.01
        self.volatility_proxy = volatility_proxy
        self.liquidity_scale = liquidity_scale

        self._l1_pool_idx = [r.l1_pool_idx for r in routes]
        self._l2_pool_idx = [r.l2_pool_idx for r in routes]
        self._bridge_idx = [r.bridge_idx for r in routes]

    @classmethod
    def from_routes(
        cls,
        snapshot: "MultiPoolSnapshot",
        routes: list["CandidateRoute"],
        **kwargs,
    ) -> "MultiPoolCostModel":
        from config.settings import BRIDGE_CONFIG

        if "latency_risk_lambda" not in kwargs:
            kwargs["latency_risk_lambda"] = BRIDGE_CONFIG.get("latency_risk_lambda", 0.01)
        return cls(routes, bridges=BRIDGE_CONFIG["bridges"], **kwargs)

    def amount_bounds_tuple(self, snapshot: "MultiPoolSnapshot") -> tuple[float, float]:
        max_eth = snapshot.inventory.max_amount_in_eth
        return 0.1, max(max_eth, 0.1)

    def estimate_gas_l1_usd(self, gas: GasState, eth_price: float) -> float:
        return self.l1_gas_limit * gas.l1_gwei * 1e-9 * eth_price

    def estimate_gas_l2_usd(self, gas: GasState, eth_price: float) -> float:
        return self.l2_gas_limit * gas.l2_gwei * 1e-9 * eth_price

    def bridge_fee_usd(self, bridge_idx: int, base_fee: float) -> float:
        if self.bridges and 0 <= bridge_idx < len(self.bridges):
            return float(self.bridges[bridge_idx].get("fee_usd", base_fee))
        return base_fee

    def bridge_latency_hours(self, bridge_idx: int) -> float:
        if self.bridges and 0 <= bridge_idx < len(self.bridges):
            return float(self.bridges[bridge_idx].get("latency_hours", 0.0))
        return 0.0

    def evaluate_route_gross_scalar(
        self,
        amount_eth: float,
        route: "CandidateRoute",
        snapshot: "MultiPoolSnapshot",
    ) -> float:
        from src.swap_simulator import estimate_cross_layer_gross_usd

        l1 = snapshot.l1_pools[route.l1_pool_idx]
        l2 = snapshot.l2_pools[route.l2_pool_idx]
        amount = min(max(amount_eth, 0.0), snapshot.inventory.max_amount_in_eth)
        return estimate_cross_layer_gross_usd(
            amount, l1, l2, liquidity_scale=self.liquidity_scale
        )

    def evaluate_route_net_scalar(
        self,
        amount_eth: float,
        route: "CandidateRoute",
        snapshot: "MultiPoolSnapshot",
    ) -> float:
        l1 = snapshot.l1_pools[route.l1_pool_idx]
        l2 = snapshot.l2_pools[route.l2_pool_idx]
        amount = min(max(amount_eth, 0.0), snapshot.inventory.max_amount_in_eth)
        gross = self.evaluate_route_gross_scalar(amount, route, snapshot)
        gas = (
            self.estimate_gas_l1_usd(snapshot.gas, l1.eth_usdc_price)
            + self.estimate_gas_l2_usd(snapshot.gas, l2.eth_usdc_price)
        )
        bridge = self.bridge_fee_usd(route.bridge_idx, snapshot.bridge_fee_usd)
        latency_risk = (
            self.latency_risk_lambda
            * self.bridge_latency_hours(route.bridge_idx)
            * amount
            * l1.eth_usdc_price
            * self.volatility_proxy
        )
        return gross - gas - bridge - latency_risk

    def _route_tensors(
        self,
        snapshot: "MultiPoolSnapshot",
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        l1_prices = []
        l2_prices = []
        l1_liq = []
        l2_liq = []
        gas_l1 = []
        gas_l2 = []
        bridge_fees = []
        latency_hours = []

        for route in self.routes:
            l1 = snapshot.l1_pools[route.l1_pool_idx]
            l2 = snapshot.l2_pools[route.l2_pool_idx]
            l1_prices.append(l1.eth_usdc_price)
            l2_prices.append(l2.eth_usdc_price)
            l1_liq.append(float(l1.liquidity))
            l2_liq.append(float(l2.liquidity))
            gas_l1.append(self.estimate_gas_l1_usd(snapshot.gas, l1.eth_usdc_price))
            gas_l2.append(self.estimate_gas_l2_usd(snapshot.gas, l2.eth_usdc_price))
            bridge_fees.append(
                self.bridge_fee_usd(route.bridge_idx, snapshot.bridge_fee_usd)
            )
            latency_hours.append(self.bridge_latency_hours(route.bridge_idx))

        return {
            "l1_prices": torch.tensor(l1_prices, device=device, dtype=dtype),
            "l2_prices": torch.tensor(l2_prices, device=device, dtype=dtype),
            "l1_liq": torch.tensor(l1_liq, device=device, dtype=dtype),
            "l2_liq": torch.tensor(l2_liq, device=device, dtype=dtype),
            "gas_total": torch.tensor(
                [g1 + g2 for g1, g2 in zip(gas_l1, gas_l2)],
                device=device,
                dtype=dtype,
            ),
            "bridge_fees": torch.tensor(bridge_fees, device=device, dtype=dtype),
            "latency_hours": torch.tensor(latency_hours, device=device, dtype=dtype),
        }

    def evaluate_profit_matrix(
        self,
        positions: torch.Tensor,
        snapshot: "MultiPoolSnapshot",
    ) -> torch.Tensor:
        """返回净利润矩阵 [N, K]（含 gas/桥费/时延惩罚）。"""
        device = positions.device
        dtype = positions.dtype
        n = positions.shape[0]
        k = self.num_candidates
        if k == 0:
            return torch.zeros(n, device=device, dtype=dtype)

        tensors = self._route_tensors(snapshot, device, dtype)
        l1_prices = tensors["l1_prices"]
        l2_prices = tensors["l2_prices"]
        l1_liq = tensors["l1_liq"]
        l2_liq = tensors["l2_liq"]

        max_eth = snapshot.inventory.max_amount_in_eth
        amount = positions[:, 0].clamp(min=0.1, max=max_eth)
        amount_nk = amount.unsqueeze(1)

        liq_scale = self.liquidity_scale
        slip1 = 1.0 / (
            1.0 + amount_nk / torch.clamp(l1_liq.unsqueeze(0) / liq_scale, min=1.0)
        )
        slip2 = 1.0 / (
            1.0 + amount_nk / torch.clamp(l2_liq.unsqueeze(0) / liq_scale, min=1.0)
        )

        fee1 = torch.tensor(
            [
                snapshot.l1_pools[r.l1_pool_idx].fee_tier / 1_000_000
                for r in self.routes
            ],
            device=device,
            dtype=dtype,
        )
        fee2 = torch.tensor(
            [
                snapshot.l2_pools[r.l2_pool_idx].fee_tier / 1_000_000
                for r in self.routes
            ],
            device=device,
            dtype=dtype,
        )

        spread = l1_prices - l2_prices
        pos_mask = spread > 0
        neg_mask = spread < 0

        buy_cost_pos = amount_nk * l2_prices.unsqueeze(0) / (
            (1.0 - fee2.unsqueeze(0)) * slip2
        )
        sell_rev_pos = amount_nk * l1_prices.unsqueeze(0) * (1.0 - fee1.unsqueeze(0)) * slip1
        gross_pos = sell_rev_pos - buy_cost_pos

        buy_cost_neg = amount_nk * l1_prices.unsqueeze(0) / (
            (1.0 - fee1.unsqueeze(0)) * slip1
        )
        sell_rev_neg = amount_nk * l2_prices.unsqueeze(0) * (1.0 - fee2.unsqueeze(0)) * slip2
        gross_neg = sell_rev_neg - buy_cost_neg

        gross = torch.zeros(n, k, device=device, dtype=dtype)
        gross = torch.where(pos_mask.unsqueeze(0), gross_pos, gross)
        gross = torch.where(neg_mask.unsqueeze(0), gross_neg, gross)

        latency_risk = (
            self.latency_risk_lambda
            * tensors["latency_hours"].unsqueeze(0)
            * amount_nk
            * l1_prices.unsqueeze(0)
            * self.volatility_proxy
        )
        net = gross - tensors["gas_total"].unsqueeze(0) - tensors["bridge_fees"].unsqueeze(0) - latency_risk
        return net

    def evaluate_fitness(
        self,
        positions: torch.Tensor,
        snapshot: "MultiPoolSnapshot",
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """粒子 fitness：profit[p, floor(cand_idx)]；可选返回 max_k 诊断。"""
        profit_matrix = self.evaluate_profit_matrix(positions, snapshot)
        k = self.num_candidates
        cand_idx = positions[:, 1].floor().long().clamp(0, max(k - 1, 0))
        fitness = profit_matrix.gather(1, cand_idx.unsqueeze(1)).squeeze(1)

        if return_diagnostics:
            max_profit = profit_matrix.max(dim=1).values
            return fitness, {"profit_matrix": profit_matrix, "max_profit": max_profit}
        return fitness


def get_multipool_search_bounds(
    snapshot: "MultiPoolSnapshot",
    num_candidates: int,
    device: str = "cpu",
) -> torch.Tensor:
    """shape [2, 2]: amount_in_eth, candidate_idx"""
    max_eth = max(snapshot.inventory.max_amount_in_eth, 0.1)
    k = max(num_candidates, 1)
    bounds = torch.tensor(
        [
            [0.1, max_eth],
            [0.0, k - 0.01],
        ],
        dtype=torch.float32,
    )
    return bounds.to(device)


def build_multipool_strategy(
    snapshot: "MultiPoolSnapshot",
    routes: list["CandidateRoute"],
    result,
) -> "ArbitrageStrategy":
    from src.types import ArbitrageStrategy

    pos = result.best_position
    amount_in = float(pos[0])
    cand_idx = int(float(pos[1]))
    cand_idx = min(max(cand_idx, 0), len(routes) - 1)
    route = routes[cand_idx]

    model = MultiPoolCostModel.from_routes(snapshot, routes)
    l1 = snapshot.l1_pools[route.l1_pool_idx]
    l2 = snapshot.l2_pools[route.l2_pool_idx]
    gas_l1 = model.estimate_gas_l1_usd(snapshot.gas, l1.eth_usdc_price)
    gas_l2 = model.estimate_gas_l2_usd(snapshot.gas, l2.eth_usdc_price)
    bridge_fee = model.bridge_fee_usd(route.bridge_idx, snapshot.bridge_fee_usd)

    return ArbitrageStrategy(
        snapshot_id=snapshot.snapshot_id,
        amount_in_eth=amount_in,
        route_l1=route.l1_pool_idx,
        route_l2=route.l2_pool_idx,
        bridge_path=route.bridge_idx,
        expected_profit_usd=result.best_fitness,
        gas_cost_usd=gas_l1 + gas_l2,
        bridge_fee_usd=bridge_fee,
        search_elapsed_ms=result.elapsed_ms,
        timestamp=snapshot.timestamp,
    )
