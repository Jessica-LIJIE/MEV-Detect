"""MEV-Detect 项目入口。

Mock 模式:  python main.py --mock
实时监听:  python main.py --live [--duration 60]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import PSO_CONFIG
from src.data_loader import load_mock_snapshots
from src.listener import MockListener, MultiChainListener
from src.models import CostModel, build_strategy, get_search_bounds
from src.optimizer import create_pso_optimizer
from src.pso_profile import get_fixed_pso_profile, select_pso_profile
from src.result_logger import append_result_log
from src.types import ArbitrageStrategy, MarketSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_pso_on_snapshot(
    snapshot: MarketSnapshot,
    *,
    adaptive: bool = False,
) -> tuple[ArbitrageStrategy, dict]:
    """对单条快照运行 PSO，返回套利策略与所用参数档位。"""
    if adaptive:
        profile = select_pso_profile(snapshot)
    else:
        profile = get_fixed_pso_profile()
        profile["spread_pct"] = round(
            (snapshot.l1.eth_usdc_price - snapshot.l2.eth_usdc_price)
            / snapshot.l2.eth_usdc_price
            * 100
            if snapshot.l2.eth_usdc_price
            else 0.0,
            4,
        )
        profile["l2_lag_ms"] = snapshot.l2_lag_ms

    device = PSO_CONFIG["device"]
    bounds = get_search_bounds(device)
    cost_model = CostModel()
    optimizer = create_pso_optimizer(
        num_particles=profile["num_particles"],
        bounds=bounds,
        device=device,
        w=PSO_CONFIG["w"],
        c1=PSO_CONFIG["c1"],
        c2=PSO_CONFIG["c2"],
        seed=42,
    )

    def fitness_fn(positions, snap=snapshot):
        return cost_model.evaluate_fitness(positions, snap)

    result = optimizer.search(fitness_fn=fitness_fn, max_iter=profile["max_iter"])
    strategy = build_strategy(snapshot, result)
    logger.info(
        "PSO profile=%s particles=%d max_iter=%d (%s)",
        profile["name"],
        profile["num_particles"],
        profile["max_iter"],
        profile["reason"],
    )
    logger.info(strategy.summary())
    logger.info(
        "  路由 L1=%d L2=%d 桥=%d | Gas $%.2f 桥费 $%.2f | 迭代 %d",
        strategy.route_l1,
        strategy.route_l2,
        strategy.bridge_path,
        strategy.gas_cost_usd,
        strategy.bridge_fee_usd,
        result.converged_at_iter,
    )
    return strategy, profile


def run_mock(record_id: str | None = None, *, adaptive: bool = False) -> None:
    snapshots = load_mock_snapshots()
    if record_id:
        snapshots = [s for s in snapshots if s.snapshot_id == record_id]
        if not snapshots:
            logger.error("未找到记录: %s", record_id)
            sys.exit(1)

    profitable = 0
    mode = "自适应" if adaptive else "固定"
    logger.info(
        "开始 Mock 回放，共 %d 条快照，设备=%s，PSO=%s",
        len(snapshots),
        PSO_CONFIG["device"],
        mode,
    )

    for snapshot in snapshots:
        if snapshot.scenario:
            logger.info("--- %s: %s ---", snapshot.snapshot_id, snapshot.scenario)

        spread = snapshot.l1.eth_usdc_price - snapshot.l2.eth_usdc_price
        logger.info(
            "L1=%.2f  L2=%.2f  价差=%.2f (%.3f%%)  L2延迟=%dms",
            snapshot.l1.eth_usdc_price,
            snapshot.l2.eth_usdc_price,
            spread,
            spread / snapshot.l2.eth_usdc_price * 100 if snapshot.l2.eth_usdc_price else 0,
            snapshot.l2_lag_ms,
        )
        strategy, profile = run_pso_on_snapshot(snapshot, adaptive=adaptive)
        append_result_log(snapshot, strategy, source="mock", pso_profile=profile)
        if strategy.expected_profit_usd > 0:
            profitable += 1

    logger.info("完成。发现套利机会 %d / %d", profitable, len(snapshots))
    logger.info("结果已追加至 data/results/live_log.jsonl")


async def run_live(
    duration_sec: float | None = None,
    use_mock_fallback: bool = True,
    use_http_poll: bool = False,
    *,
    adaptive: bool = False,
) -> None:
    if not MultiChainListener.is_configured() and not MultiChainListener.is_http_configured():
        if use_mock_fallback:
            logger.warning("未配置 WebSocket RPC，改用 Mock 监听器回放")
            listener = MockListener()

            async def on_snapshot(snap: MarketSnapshot) -> None:
                strategy, profile = run_pso_on_snapshot(snap, adaptive=adaptive)
                append_result_log(snap, strategy, source="mock", pso_profile=profile)

            listener.on_snapshot(on_snapshot)
            await listener.replay(interval_sec=0.3)
            return
        logger.error("请在 .env 中配置 ETH_WS_URL 与 ARB_WS_URL")
        sys.exit(1)

    listener = MultiChainListener(use_http_poll=use_http_poll)

    async def on_snapshot(snapshot: MarketSnapshot) -> None:
        spread = snapshot.l1.eth_usdc_price - snapshot.l2.eth_usdc_price
        logger.info(
            "收到快照 | L1 block=%d (%.2f) | L2 block=%d (%.2f) | 价差=%.2f",
            snapshot.l1.block_number,
            snapshot.l1.eth_usdc_price,
            snapshot.l2.block_number,
            snapshot.l2.eth_usdc_price,
            spread,
        )
        strategy, profile = run_pso_on_snapshot(snapshot, adaptive=adaptive)
        append_result_log(snapshot, strategy, source="live", pso_profile=profile)

    listener.on_snapshot(on_snapshot)

    if duration_sec:
        logger.info("实时监听 %g 秒...", duration_sec)
        await listener.run_for(duration_sec)
    else:
        logger.info("实时监听已启动，按 Ctrl+C 停止")
        await listener.start()
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            await listener.stop()


def main():
    parser = argparse.ArgumentParser(description="L1-L2 跨层 MEV 检测框架")
    parser.add_argument("--mock", action="store_true", help="使用 mock_mempool.json 离线回放")
    parser.add_argument("--live", action="store_true", help="实时双链监听")
    parser.add_argument(
        "--http",
        action="store_true",
        help="使用 HTTP 轮询（WebSocket 连不上时推荐，如 ETH 主网超时）",
    )
    parser.add_argument("--duration", type=float, default=None, help="实时监听时长（秒）")
    parser.add_argument("--record-id", type=str, default=None, help="仅运行指定快照 ID")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="启用 L1/L2 自适应 PSO 参数（E3c）",
    )
    args = parser.parse_args()

    if args.live:
        asyncio.run(
            run_live(
                duration_sec=args.duration,
                use_http_poll=args.http,
                adaptive=args.adaptive,
            )
        )
    else:
        run_mock(record_id=args.record_id, adaptive=args.adaptive)


if __name__ == "__main__":
    main()
