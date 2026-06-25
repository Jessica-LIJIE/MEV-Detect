"""L1/L2 双链异步监听器：订阅 Swap 事件，大额 L1 交易触发快照回调。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from web3 import AsyncWeb3, Web3
from web3.providers.persistent import WebSocketProvider
from web3.utils.subscriptions import LogsSubscription, NewHeadsSubscription

from config.settings import COST_DEFAULTS, LISTENER_CONFIG, POOLS, RPC
from src.rpc_utils import create_sync_w3, get_websocket_kwargs
from src.pool_utils import (
    POOL_ABI,
    SWAP_TOPIC,
    decode_swap_log,
    estimate_swap_usd,
    sqrt_price_x96_to_eth_usdc,
)
from src.types import GasState, MarketSnapshot, PoolState

logger = logging.getLogger(__name__)

SnapshotCallback = Callable[[MarketSnapshot], Awaitable[None]]


@dataclass
class ChainState:
    chain: str
    pool_address: str
    block_number: int = 0
    block_timestamp: int = 0
    block_time: str = ""
    sqrt_price_x96: int = 0
    liquidity: int = 0
    tick: int = 0
    eth_usdc_price: float = 0.0
    gas_gwei: float = 0.0


@dataclass
class MultiChainListener:
    """同时监听 Ethereum (L1) 与 Arbitrum (L2) 的区块头与 Swap 事件。"""

    swap_threshold_usd: float = field(
        default_factory=lambda: LISTENER_CONFIG["l1_swap_threshold_usd"]
    )
    reconnect_delay_sec: float = field(
        default_factory=lambda: LISTENER_CONFIG["reconnect_delay_sec"]
    )
    max_reconnect_attempts: int = field(
        default_factory=lambda: LISTENER_CONFIG["max_reconnect_attempts"]
    )
    bridge_fee_usd: float = field(
        default_factory=lambda: COST_DEFAULTS["bridge_fee_usd"]
    )
    http_poll_interval: float = 3.0
    use_http_poll: bool = False

    l1_state: ChainState = field(init=False)
    l2_state: ChainState = field(init=False)
    _callbacks: list[SnapshotCallback] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _snapshot_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.l1_state = ChainState(
            chain="ethereum",
            pool_address=Web3.to_checksum_address(POOLS["ethereum"]["eth_usdc_005"]),
        )
        self.l2_state = ChainState(
            chain="arbitrum",
            pool_address=Web3.to_checksum_address(POOLS["arbitrum"]["eth_usdc_005"]),
        )

    def on_snapshot(self, callback: SnapshotCallback) -> None:
        self._callbacks.append(callback)

    @staticmethod
    def is_configured() -> bool:
        return bool(RPC["ethereum"]["ws"] and RPC["arbitrum"]["ws"])

    @staticmethod
    def is_http_configured() -> bool:
        return bool(RPC["ethereum"]["http"] and RPC["arbitrum"]["http"])

    @staticmethod
    def _timestamp_to_iso(ts: int) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fetch_pool_state_sync(
        self, http_url: str, pool_address: str, chain: str
    ) -> tuple[int, int, int, float]:
        w3 = create_sync_w3(http_url)
        contract = w3.eth.contract(address=pool_address, abi=POOL_ABI)
        slot0 = contract.functions.slot0().call()
        liq = contract.functions.liquidity().call()
        sqrt_price_x96 = int(slot0[0])
        tick = int(slot0[1])
        liquidity = int(liq)
        price = sqrt_price_x96_to_eth_usdc(sqrt_price_x96, chain)
        return sqrt_price_x96, liquidity, tick, price

    def _update_chain_state_from_http(self, state: ChainState, http_url: str) -> None:
        sqrt_price_x96, liquidity, tick, price = self._fetch_pool_state_sync(
            http_url, state.pool_address, state.chain
        )
        state.sqrt_price_x96 = sqrt_price_x96
        state.liquidity = liquidity
        state.tick = tick
        state.eth_usdc_price = price

    def _sync_fetch_block_and_gas(self, http_url: str) -> tuple[dict, float]:
        w3 = create_sync_w3(http_url)
        block = w3.eth.get_block("latest")
        gas_gwei = float(Web3.from_wei(w3.eth.gas_price, "gwei"))
        return block, gas_gwei

    async def _handle_new_head(self, chain: str, block: dict) -> None:
        block_number = int(block["number"])
        block_timestamp = int(block["timestamp"])
        block_time = self._timestamp_to_iso(block_timestamp)

        http_url = RPC[chain]["http"]
        _, gas_gwei = await asyncio.to_thread(self._sync_fetch_block_and_gas, http_url)

        state = self.l1_state if chain == "ethereum" else self.l2_state
        async with self._lock:
            state.block_number = block_number
            state.block_timestamp = block_timestamp
            state.block_time = block_time
            state.gas_gwei = gas_gwei

        if state.sqrt_price_x96 == 0:
            await asyncio.to_thread(self._update_chain_state_from_http, state, http_url)

        logger.debug(
            "[%s] newHead block=%d price=%.2f gas=%.2f gwei",
            chain,
            block_number,
            state.eth_usdc_price,
            gas_gwei,
        )

    async def _handle_l1_swap(self, log: dict) -> None:
        swap = decode_swap_log(log)
        eth_price = sqrt_price_x96_to_eth_usdc(swap.sqrt_price_x96, "ethereum")
        swap_usd = estimate_swap_usd(swap.amount0, swap.amount1, eth_price, "ethereum")

        async with self._lock:
            self.l1_state.sqrt_price_x96 = swap.sqrt_price_x96
            self.l1_state.liquidity = swap.liquidity
            self.l1_state.tick = swap.tick
            self.l1_state.eth_usdc_price = eth_price
            if swap.block_number:
                self.l1_state.block_number = swap.block_number

        logger.info(
            "[L1 Swap] block=%s amount_usd≈%.0f price=%.2f tx=%s",
            swap.block_number,
            swap_usd,
            eth_price,
            swap.tx_hash[:18],
        )

        if swap_usd < self.swap_threshold_usd:
            return

        await self._refresh_l2_state()
        snapshot = await self._build_snapshot(
            trigger="l1_large_swap",
            swap_amount_usd=swap_usd,
        )
        await self._emit_snapshot(snapshot)

    async def _handle_l2_swap(self, log: dict) -> None:
        swap = decode_swap_log(log)
        eth_price = sqrt_price_x96_to_eth_usdc(swap.sqrt_price_x96, "arbitrum")

        async with self._lock:
            self.l2_state.sqrt_price_x96 = swap.sqrt_price_x96
            self.l2_state.liquidity = swap.liquidity
            self.l2_state.tick = swap.tick
            self.l2_state.eth_usdc_price = eth_price
            if swap.block_number:
                self.l2_state.block_number = swap.block_number

        logger.debug(
            "[L2 Swap] block=%s price=%.2f",
            swap.block_number,
            eth_price,
        )

    async def _refresh_l2_state(self) -> None:
        http_url = RPC["arbitrum"]["http"]
        if not http_url:
            return

        def _fetch():
            self._update_chain_state_from_http(self.l2_state, http_url)
            block, gas_gwei = self._sync_fetch_block_and_gas(http_url)
            self.l2_state.block_number = int(block["number"])
            self.l2_state.block_timestamp = int(block["timestamp"])
            self.l2_state.block_time = self._timestamp_to_iso(int(block["timestamp"]))
            self.l2_state.gas_gwei = gas_gwei

        await asyncio.to_thread(_fetch)

    def _pool_state_from_chain(self, state: ChainState) -> PoolState:
        return PoolState(
            chain=state.chain,
            block_number=state.block_number,
            block_time=state.block_time or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            pool_address=state.pool_address,
            sqrt_price_x96=state.sqrt_price_x96,
            liquidity=state.liquidity,
            tick=state.tick,
            eth_usdc_price=state.eth_usdc_price,
        )

    async def _build_snapshot(
        self,
        trigger: str,
        swap_amount_usd: Optional[float] = None,
    ) -> MarketSnapshot:
        async with self._lock:
            l1 = self.l1_state
            l2 = self.l2_state
            self._snapshot_counter += 1

            if l1.block_timestamp and l2.block_timestamp:
                l2_lag_ms = max(0, (l2.block_timestamp - l1.block_timestamp) * 1000)
            else:
                l2_lag_ms = 0

            snapshot_id = f"live_{self._snapshot_counter:04d}"

        return MarketSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            trigger=trigger,
            l1=self._pool_state_from_chain(l1),
            l2=self._pool_state_from_chain(l2),
            l2_lag_ms=int(l2_lag_ms),
            gas=GasState(l1_gwei=l1.gas_gwei, l2_gwei=l2.gas_gwei),
            bridge_fee_usd=self.bridge_fee_usd,
            swap_amount_usd=swap_amount_usd,
        )

    async def _emit_snapshot(self, snapshot: MarketSnapshot) -> None:
        logger.info(
            "触发快照 %s | L1 block=%d price=%.2f | L2 block=%d price=%.2f | lag=%dms",
            snapshot.snapshot_id,
            snapshot.l1.block_number,
            snapshot.l1.eth_usdc_price,
            snapshot.l2.block_number,
            snapshot.l2.eth_usdc_price,
            snapshot.l2_lag_ms,
        )
        for callback in self._callbacks:
            await callback(snapshot)

    async def get_latest_snapshot(self) -> MarketSnapshot:
        """拉取链上最新池子状态并组装快照（用于连通性测试）。"""
        if not RPC["ethereum"]["http"] or not RPC["arbitrum"]["http"]:
            raise RuntimeError("请在 .env 中配置 ETH_HTTP_URL 与 ARB_HTTP_URL")

        def _fetch_all():
            for http_url, state in (
                (RPC["ethereum"]["http"], self.l1_state),
                (RPC["arbitrum"]["http"], self.l2_state),
            ):
                self._update_chain_state_from_http(state, http_url)
                block, gas_gwei = self._sync_fetch_block_and_gas(http_url)
                state.block_number = int(block["number"])
                state.block_timestamp = int(block["timestamp"])
                state.block_time = self._timestamp_to_iso(int(block["timestamp"]))
                state.gas_gwei = gas_gwei

        await asyncio.to_thread(_fetch_all)
        return await self._build_snapshot(trigger="manual_fetch")

    @staticmethod
    def _normalize_log(log: dict) -> dict:
        data = log.get("data", "0x")
        if hasattr(data, "hex"):
            data = data.hex()
        if isinstance(data, bytes):
            data = "0x" + data.hex()
        tx_hash = log.get("transactionHash", "")
        if hasattr(tx_hash, "hex"):
            tx_hash = tx_hash.hex()
        return {
            "data": data if str(data).startswith("0x") else "0x" + str(data),
            "blockNumber": int(log["blockNumber"]),
            "transactionHash": tx_hash,
        }

    def _poll_chain_sync(self, chain: str, last_block: int) -> tuple[int, list[dict], dict | None]:
        http_url = RPC[chain]["http"]
        state = self.l1_state if chain == "ethereum" else self.l2_state
        w3 = create_sync_w3(http_url)
        latest = w3.eth.get_block("latest")
        latest_num = int(latest["number"])
        logs: list[dict] = []
        if latest_num > last_block:
            raw_logs = w3.eth.get_logs(
                {
                    "address": state.pool_address,
                    "topics": [SWAP_TOPIC],
                    "fromBlock": last_block + 1,
                    "toBlock": latest_num,
                }
            )
            logs = [self._normalize_log(dict(log)) for log in raw_logs]
            return latest_num, logs, dict(latest)
        return last_block, logs, None

    async def _run_chain_http_poll(self, chain: str) -> None:
        """HTTP 轮询模式：WebSocket 不稳定时（如 ETH 主网）的可靠替代方案。"""
        http_url = RPC[chain]["http"]
        if not http_url:
            logger.error("[%s] 未配置 HTTP URL，无法轮询", chain)
            return

        is_l1 = chain == "ethereum"
        block, _ = await asyncio.to_thread(self._sync_fetch_block_and_gas, http_url)
        await self._handle_new_head(chain, block)
        last_block = int(block["number"])

        logger.info(
            "[%s] HTTP 轮询已启动，间隔 %.1fs，池子 %s",
            chain,
            self.http_poll_interval,
            self.l1_state.pool_address if is_l1 else self.l2_state.pool_address,
        )

        while self._running:
            try:
                latest_num, logs, latest_block = await asyncio.to_thread(
                    self._poll_chain_sync, chain, last_block
                )
                if latest_block is not None:
                    await self._handle_new_head(chain, latest_block)
                    for log in logs:
                        if is_l1:
                            await self._handle_l1_swap(log)
                        else:
                            await self._handle_l2_swap(log)
                    last_block = latest_num
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[%s] HTTP 轮询异常: %s", chain, exc)
            await asyncio.sleep(self.http_poll_interval)

    async def _run_chain_session(self, chain: str) -> None:
        ws_url = RPC[chain]["ws"]
        state = self.l1_state if chain == "ethereum" else self.l2_state
        is_l1 = chain == "ethereum"

        http_url = RPC[chain]["http"]
        await asyncio.to_thread(self._update_chain_state_from_http, state, http_url)
        block, _ = await asyncio.to_thread(self._sync_fetch_block_and_gas, http_url)
        await self._handle_new_head(chain, block)

        async with AsyncWeb3(
            WebSocketProvider(ws_url, websocket_kwargs=get_websocket_kwargs())
        ) as w3:

            async def on_new_head(ctx) -> None:
                if not self._running:
                    await ctx.subscription.unsubscribe()
                    return
                await self._handle_new_head(chain, ctx.result)

            async def on_swap_log(ctx) -> None:
                if not self._running:
                    await ctx.subscription.unsubscribe()
                    return
                if is_l1:
                    await self._handle_l1_swap(ctx.result)
                else:
                    await self._handle_l2_swap(ctx.result)

            subs = [
                NewHeadsSubscription(label=f"{chain}_heads", handler=on_new_head),
                LogsSubscription(
                    address=state.pool_address,
                    topics=[SWAP_TOPIC],
                    label=f"{chain}_swaps",
                    handler=on_swap_log,
                ),
            ]
            await w3.subscription_manager.subscribe(subs)
            logger.info("[%s] WebSocket 已连接，监听池子 %s", chain, state.pool_address)
            await w3.subscription_manager.handle_subscriptions(run_forever=True)

    async def _chain_loop(self, chain: str) -> None:
        if self.use_http_poll:
            await self._run_chain_http_poll(chain)
            return

        delay = self.reconnect_delay_sec
        attempts = 0

        while self._running:
            try:
                await self._run_chain_session(chain)
                if not self._running:
                    break
                attempts = 0
                delay = self.reconnect_delay_sec
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts += 1
                logger.warning(
                    "[%s] WebSocket 异常: %s | %d 次重试",
                    chain,
                    exc,
                    attempts,
                )
                if attempts >= 2 and self.is_http_configured():
                    logger.warning(
                        "[%s] WebSocket 不可用，自动切换 HTTP 轮询模式",
                        chain,
                    )
                    await self._run_chain_http_poll(chain)
                    return
                if self.max_reconnect_attempts > 0 and attempts >= self.max_reconnect_attempts:
                    if self.is_http_configured():
                        logger.warning("[%s] 改用 HTTP 轮询模式", chain)
                        await self._run_chain_http_poll(chain)
                    else:
                        logger.error("[%s] 达到最大重连次数，停止监听", chain)
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def start(self) -> None:
        if self.use_http_poll:
            if not self.is_http_configured():
                raise RuntimeError("HTTP 轮询模式需要 ETH_HTTP_URL 与 ARB_HTTP_URL")
        elif not self.is_configured():
            if self.is_http_configured():
                logger.warning("未配置 WebSocket，自动使用 HTTP 轮询模式")
                self.use_http_poll = True
            else:
                raise RuntimeError(
                    "请配置 RPC：至少设置 ETH_HTTP_URL 与 ARB_HTTP_URL"
                )
        if self._running:
            return

        self._running = True
        mode = "HTTP 轮询" if self.use_http_poll else "WebSocket"
        self._tasks = [
            asyncio.create_task(self._chain_loop("ethereum"), name="listener-ethereum"),
            asyncio.create_task(self._chain_loop("arbitrum"), name="listener-arbitrum"),
        ]
        logger.info("双链监听器已启动（%s）", mode)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("双链监听器已停止")

    async def run_for(self, duration_sec: float) -> None:
        """运行指定秒数后自动停止。"""
        await self.start()
        try:
            await asyncio.sleep(duration_sec)
        finally:
            await self.stop()


class MockListener:
    """无 RPC 时回放 mock_mempool.json，模拟监听器回调。"""

    def __init__(self):
        self._callbacks: list[SnapshotCallback] = []

    def on_snapshot(self, callback: SnapshotCallback) -> None:
        self._callbacks.append(callback)

    async def replay(self, record_id: str | None = None, interval_sec: float = 0.5) -> int:
        from src.data_loader import load_mock_snapshots

        snapshots = load_mock_snapshots()
        if record_id:
            snapshots = [s for s in snapshots if s.snapshot_id == record_id]

        for snapshot in snapshots:
            logger.info("Mock 回放: %s", snapshot.snapshot_id)
            for callback in self._callbacks:
                await callback(snapshot)
            if interval_sec > 0:
                await asyncio.sleep(interval_sec)
        return len(snapshots)
