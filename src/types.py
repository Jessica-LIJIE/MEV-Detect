from dataclasses import dataclass
from typing import Optional


@dataclass
class InventoryState:
    """分层库存约束：L1/L2 各自可动用的 ETH 上限。"""

    l1_eth: float
    l2_eth: float

    @property
    def max_amount_in_eth(self) -> float:
        return min(self.l1_eth, self.l2_eth)


@dataclass
class PoolSnapshot:
    """多池快照中单池链上状态（Phase 1 RPC 采集填充）。"""

    chain: str
    pool_address: str
    token0: str
    token1: str
    fee_tier: int
    sqrt_price_x96: int
    liquidity: int
    tick: int
    eth_usdc_price: float
    ticks_loaded: bool = False
    pool_id: str = ""
    block_number: int = 0
    block_time: str = ""
    initialized_ticks: list[dict] | None = None


@dataclass
class MultiPoolSnapshot:
    """跨层多池一致视图（真实 RPC 或 Mock multipool）。"""

    snapshot_id: str
    l1_block: int
    l2_block: int
    l1_pools: list[PoolSnapshot]
    l2_pools: list[PoolSnapshot]
    gas: "GasState"
    bridge_fee_usd: float
    inventory: InventoryState
    trigger: str
    timestamp: str
    l1_batch: int | None = None
    l2_lag_ms: int = 0
    swap_amount_usd: float | None = None
    scenario: str | None = None

    @property
    def num_l1_pools(self) -> int:
        return len(self.l1_pools)

    @property
    def num_l2_pools(self) -> int:
        return len(self.l2_pools)


@dataclass
class CandidateRoute:
    """阶段1 输出的跨层候选路由（MVP：L1池 × 桥 × L2池）。"""

    route_id: int
    l1_pool_idx: int
    l2_pool_idx: int
    bridge_idx: int
    score: float
    l1_pool_id: str = ""
    l2_pool_id: str = ""
    bridge_id: str = ""


@dataclass
class PoolState:
    chain: str
    block_number: int
    block_time: str
    pool_address: str
    sqrt_price_x96: int
    liquidity: int
    tick: int
    eth_usdc_price: float


@dataclass
class GasState:
    l1_gwei: float
    l2_gwei: float


@dataclass
class MarketSnapshot:
    snapshot_id: str
    timestamp: str
    trigger: str
    l1: PoolState
    l2: PoolState
    l2_lag_ms: int
    gas: GasState
    bridge_fee_usd: float
    swap_amount_usd: Optional[float] = None
    scenario: Optional[str] = None


@dataclass
class SearchResult:
    best_position: list[float]
    best_fitness: float
    converged_at_iter: int
    elapsed_ms: float
    fitness_history: list[float]


@dataclass
class ArbitrageStrategy:
    snapshot_id: str
    amount_in_eth: float
    route_l1: int
    route_l2: int
    bridge_path: int
    expected_profit_usd: float
    gas_cost_usd: float
    bridge_fee_usd: float
    search_elapsed_ms: float
    timestamp: str

    def summary(self) -> str:
        status = "有机会" if self.expected_profit_usd > 0 else "无套利"
        return (
            f"[{self.snapshot_id}] {status} | "
            f"投入 {self.amount_in_eth:.2f} ETH | "
            f"净利润 ${self.expected_profit_usd:.2f} | "
            f"耗时 {self.search_elapsed_ms:.1f} ms"
        )
