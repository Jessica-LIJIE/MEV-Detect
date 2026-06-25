"""Fork / QuoterV2 路由验真（Phase 5）。

L1：优先 anvil fork；不可用时回退至 RPC 钉块 QuoterV2。
L2：Arbitrum RPC 钉块 QuoterV2（MVP 不做 L2 anvil）。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from web3 import Web3

from config.settings import BRIDGE_CONFIG, RPC, ROOT
from src.models import MultiPoolCostModel
from src.pool_registry import PoolRegistryEntry, get_contract_addresses, load_registry, load_registry_raw
from src.rpc_utils import create_sync_w3, authenticated_fork_url
from src.types import CandidateRoute, MultiPoolSnapshot, PoolSnapshot

logger = logging.getLogger(__name__)

FORK_VERIFY_LOG = ROOT / "data" / "results" / "fork_verify.jsonl"

# Foundry anvil 默认账户 #0（EIP-55 checksum；anvil 默认解锁）
ANVIL_DEFAULT_ACCOUNT = "0xf39Fd6e51aad88F6F4ce6b882Fc0C556c753c757"

ERC20_ABI = [
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

WETH_ABI = ERC20_ABI + [
    {
        "inputs": [],
        "name": "deposit",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
]

SWAP_ROUTER02_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "recipient", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMinimum", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
                "name": "params",
                "type": "tuple",
            }
        ],
        "name": "exactInputSingle",
        "outputs": [{"name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
        "type": "function",
    },
]

QUOTER_V2_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
                "name": "params",
                "type": "tuple",
            }
        ],
        "name": "quoteExactInputSingle",
        "outputs": [
            {"name": "amountOut", "type": "uint256"},
            {"name": "sqrtPriceX96After", "type": "uint160"},
            {"name": "initializedTicksCrossed", "type": "uint32"},
            {"name": "gasEstimate", "type": "uint256"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "amountOut", "type": "uint256"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
                "name": "params",
                "type": "tuple",
            }
        ],
        "name": "quoteExactOutputSingle",
        "outputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "sqrtPriceX96After", "type": "uint160"},
            {"name": "initializedTicksCrossed", "type": "uint32"},
            {"name": "gasEstimate", "type": "uint256"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass
class ForkVerifyResult:
    snapshot_id: str
    route_id: int
    l1_pool_id: str
    l2_pool_id: str
    bridge_id: str
    amount_in_eth: float
    l1_block: int
    l2_block: int
    simulated_profit: float
    fork_profit: float
    rel_error: float
    verification_mode: str
    quoter_ok: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def registry_entry_for_pool(pool: PoolSnapshot) -> PoolRegistryEntry:
    for entry in load_registry(pool.chain):
        if entry.id == pool.pool_id:
            return entry
    raise KeyError(f"Pool {pool.pool_id} not found in registry for chain {pool.chain}")


def pool_weth_stable(entry: PoolRegistryEntry) -> tuple[str, str, int, int]:
    """返回 (weth_address, stable_address, weth_decimals, stable_decimals)。"""
    if entry.weth_is_token0:
        return (
            entry.token0_address,
            entry.token1_address,
            entry.token0_decimals,
            entry.token1_decimals,
        )
    return (
        entry.token1_address,
        entry.token0_address,
        entry.token1_decimals,
        entry.token0_decimals,
    )


def eth_to_wei(amount_eth: float) -> int:
    return int(max(amount_eth, 0.0) * 10**18)


def stable_to_usd(amount_wei: int, stable_decimals: int) -> float:
    return float(amount_wei) / (10**stable_decimals)


def rel_error(simulated: float, fork: float) -> float:
    denom = max(abs(simulated), abs(fork), 1e-9)
    return abs(simulated - fork) / denom


def _quoter_contract(w3: Web3, chain: str):
    addrs = get_contract_addresses(chain)
    quoter = addrs.get("quoter_v2")
    if not quoter:
        raise ValueError(f"QuoterV2 missing for chain {chain}")
    return w3.eth.contract(
        address=Web3.to_checksum_address(quoter),
        abi=QUOTER_V2_ABI,
    )


def quoter_exact_input(
    w3: Web3,
    chain: str,
    token_in: str,
    token_out: str,
    amount_in_wei: int,
    fee_tier: int,
    *,
    block_identifier: int,
) -> int | None:
    try:
        contract = _quoter_contract(w3, chain)
        params = (
            Web3.to_checksum_address(token_in),
            Web3.to_checksum_address(token_out),
            int(amount_in_wei),
            int(fee_tier),
            0,
        )
        amount_out, _, _, _ = contract.functions.quoteExactInputSingle(params).call(
            block_identifier=block_identifier
        )
        return int(amount_out)
    except Exception as exc:
        logger.debug("quoteExactInputSingle failed: %s", exc)
        return None


def quoter_exact_output(
    w3: Web3,
    chain: str,
    token_in: str,
    token_out: str,
    amount_out_wei: int,
    fee_tier: int,
    *,
    block_identifier: int,
) -> int | None:
    try:
        contract = _quoter_contract(w3, chain)
        params = (
            Web3.to_checksum_address(token_in),
            Web3.to_checksum_address(token_out),
            int(amount_out_wei),
            int(fee_tier),
            0,
        )
        amount_in, _, _, _ = contract.functions.quoteExactOutputSingle(params).call(
            block_identifier=block_identifier
        )
        return int(amount_in)
    except Exception as exc:
        logger.debug("quoteExactOutputSingle failed: %s", exc)
        return None


def quote_cross_layer_gross_usd(
    w3_l1: Web3,
    w3_l2: Web3,
    snapshot: MultiPoolSnapshot,
    route: CandidateRoute,
    amount_eth: float,
) -> tuple[float | None, str]:
    """QuoterV2 两腿报价毛利润（USD）。返回 (gross_usd, note)。"""
    l1_pool = snapshot.l1_pools[route.l1_pool_idx]
    l2_pool = snapshot.l2_pools[route.l2_pool_idx]
    l1_entry = registry_entry_for_pool(l1_pool)
    l2_entry = registry_entry_for_pool(l2_pool)

    weth_l1, stable_l1, _, stable_dec_l1 = pool_weth_stable(l1_entry)
    weth_l2, stable_l2, _, stable_dec_l2 = pool_weth_stable(l2_entry)
    amount_wei = eth_to_wei(amount_eth)
    if amount_wei <= 0:
        return 0.0, "zero amount"

    spread = l1_pool.eth_usdc_price - l2_pool.eth_usdc_price
    if spread > 0:
        buy_stable_wei = quoter_exact_output(
            w3_l2,
            "arbitrum",
            stable_l2,
            weth_l2,
            amount_wei,
            l2_pool.fee_tier,
            block_identifier=snapshot.l2_block,
        )
        sell_stable_wei = quoter_exact_input(
            w3_l1,
            "ethereum",
            weth_l1,
            stable_l1,
            amount_wei,
            l1_pool.fee_tier,
            block_identifier=snapshot.l1_block,
        )
        if buy_stable_wei is None or sell_stable_wei is None:
            return None, "quoter failed on buy-L2 or sell-L1 leg"
        buy_usd = stable_to_usd(buy_stable_wei, stable_dec_l2)
        sell_usd = stable_to_usd(sell_stable_wei, stable_dec_l1)
        return sell_usd - buy_usd, "L1 sell / L2 buy"

    if spread < 0:
        buy_stable_wei = quoter_exact_output(
            w3_l1,
            "ethereum",
            stable_l1,
            weth_l1,
            amount_wei,
            l1_pool.fee_tier,
            block_identifier=snapshot.l1_block,
        )
        sell_stable_wei = quoter_exact_input(
            w3_l2,
            "arbitrum",
            weth_l2,
            stable_l2,
            amount_wei,
            l2_pool.fee_tier,
            block_identifier=snapshot.l2_block,
        )
        if buy_stable_wei is None or sell_stable_wei is None:
            return None, "quoter failed on buy-L1 or sell-L2 leg"
        buy_usd = stable_to_usd(buy_stable_wei, stable_dec_l1)
        sell_usd = stable_to_usd(sell_stable_wei, stable_dec_l2)
        return sell_usd - buy_usd, "L1 buy / L2 sell"

    return 0.0, "zero spread"


def fork_net_profit_from_gross(
    gross_usd: float,
    snapshot: MultiPoolSnapshot,
    route: CandidateRoute,
    amount_eth: float,
    model: MultiPoolCostModel,
) -> float:
    l1 = snapshot.l1_pools[route.l1_pool_idx]
    l2 = snapshot.l2_pools[route.l2_pool_idx]
    gas = model.estimate_gas_l1_usd(snapshot.gas, l1.eth_usdc_price) + model.estimate_gas_l2_usd(
        snapshot.gas, l2.eth_usdc_price
    )
    bridge = model.bridge_fee_usd(route.bridge_idx, snapshot.bridge_fee_usd)
    latency = (
        model.latency_risk_lambda
        * model.bridge_latency_hours(route.bridge_idx)
        * amount_eth
        * l1.eth_usdc_price
        * model.volatility_proxy
    )
    return gross_usd - gas - bridge - latency


def _router_contract(w3: Web3, chain: str):
    addrs = get_contract_addresses(chain)
    router = addrs.get("swap_router02")
    if not router:
        raise ValueError(f"SwapRouter02 missing for chain {chain}")
    return w3.eth.contract(
        address=Web3.to_checksum_address(router),
        abi=SWAP_ROUTER02_ABI,
    )


def anvil_set_balance(w3: Web3, account: str, wei: int) -> None:
    w3.provider.make_request("anvil_setBalance", [Web3.to_checksum_address(account), hex(wei)])


def _send_anvil_transaction(w3: Web3, tx: dict) -> str:
    """通过 anvil 解锁账户发送交易（无需本地私钥）。"""
    payload = dict(tx)
    from_addr = Web3.to_checksum_address(payload["from"])
    payload["from"] = from_addr
    if "to" in payload:
        payload["to"] = Web3.to_checksum_address(payload["to"])
    w3.provider.make_request("anvil_impersonateAccount", [from_addr])
    for field in ("value", "gas", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas", "nonce", "chainId"):
        if field in payload and not isinstance(payload[field], str):
            payload[field] = hex(payload[field])
    result = w3.provider.make_request("eth_sendTransaction", [payload])
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["result"]


def fund_weth_on_fork(
    w3: Web3,
    account: str,
    amount_wei: int,
    weth_address: str,
) -> None:
    """在 fork 上为账户准备 WETH（ETH deposit）；anvil 默认账户已解锁。"""
    gas_buffer = w3.to_wei(0.5, "ether")
    anvil_set_balance(w3, account, amount_wei + gas_buffer)
    weth = w3.eth.contract(address=Web3.to_checksum_address(weth_address), abi=WETH_ABI)
    acct = Web3.to_checksum_address(account)
    gas_price = int(w3.eth.gas_price)
    tx = weth.functions.deposit().build_transaction(
        {
            "from": acct,
            "value": amount_wei,
            "nonce": w3.eth.get_transaction_count(acct),
            "gas": 120_000,
            "gasPrice": gas_price,
            "chainId": w3.eth.chain_id,
        }
    )
    tx_hash = _send_anvil_transaction(w3, tx)
    w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)


def execute_l1_exact_input_swap(
    w3: Web3,
    *,
    token_in: str,
    token_out: str,
    fee_tier: int,
    amount_in_wei: int,
    recipient: str,
) -> tuple[int | None, str]:
    """在 L1 anvil fork 上执行 SwapRouter02.exactInputSingle，返回 (amountOut, note)。"""
    if amount_in_wei <= 0:
        return 0, "zero amount"

    account = Web3.to_checksum_address(recipient)
    router = _router_contract(w3, "ethereum")
    token_in_c = Web3.to_checksum_address(token_in)
    token_out_c = Web3.to_checksum_address(token_out)
    gas_price = int(w3.eth.gas_price)

    weth_addr = Web3.to_checksum_address(load_registry_raw()["tokens"]["ethereum"]["WETH"])
    if token_in_c.lower() == weth_addr.lower():
        fund_weth_on_fork(w3, account, amount_in_wei, weth_addr)
        token_contract = w3.eth.contract(address=weth_addr, abi=WETH_ABI)
    else:
        token_contract = w3.eth.contract(address=token_in_c, abi=ERC20_ABI)

    stable = w3.eth.contract(address=token_out_c, abi=ERC20_ABI)
    bal_before = stable.functions.balanceOf(account).call()

    router_addr = router.address
    approve_tx = token_contract.functions.approve(router_addr, amount_in_wei).build_transaction(
        {
            "from": account,
            "nonce": w3.eth.get_transaction_count(account),
            "gas": 100_000,
            "gasPrice": gas_price,
            "chainId": w3.eth.chain_id,
        }
    )
    approve_hash = _send_anvil_transaction(w3, approve_tx)
    w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)

    params = (
        token_in_c,
        token_out_c,
        int(fee_tier),
        account,
        int(amount_in_wei),
        0,
        0,
    )
    swap_tx = router.functions.exactInputSingle(params).build_transaction(
        {
            "from": account,
            "nonce": w3.eth.get_transaction_count(account),
            "gas": 500_000,
            "gasPrice": gas_price,
            "chainId": w3.eth.chain_id,
        }
    )
    swap_hash = _send_anvil_transaction(w3, swap_tx)
    receipt = w3.eth.wait_for_transaction_receipt(swap_hash, timeout=120)
    if receipt["status"] != 1:
        return None, "swap transaction reverted"

    bal_after = stable.functions.balanceOf(account).call()
    amount_out = int(bal_after - bal_before)
    if amount_out <= 0:
        return None, "swap produced zero output"
    return amount_out, "L1 exactInputSingle executed on anvil fork"


def quote_cross_layer_gross_with_l1_swap(
    w3_l1_fork: Web3,
    w3_l2: Web3,
    snapshot: MultiPoolSnapshot,
    route: CandidateRoute,
    amount_eth: float,
    *,
    swap_account: str = ANVIL_DEFAULT_ACCOUNT,
) -> tuple[float | None, str]:
    """L1 腿 state-changing swap + L2 腿 QuoterV2（跨层验真推荐组合）。"""
    l1_pool = snapshot.l1_pools[route.l1_pool_idx]
    l2_pool = snapshot.l2_pools[route.l2_pool_idx]
    l1_entry = registry_entry_for_pool(l1_pool)
    l2_entry = registry_entry_for_pool(l2_pool)

    weth_l1, stable_l1, _, stable_dec_l1 = pool_weth_stable(l1_entry)
    weth_l2, stable_l2, _, stable_dec_l2 = pool_weth_stable(l2_entry)
    amount_wei = eth_to_wei(amount_eth)
    if amount_wei <= 0:
        return 0.0, "zero amount"

    spread = l1_pool.eth_usdc_price - l2_pool.eth_usdc_price
    if spread <= 0:
        return None, "L1 swap exec only implemented for spread>0 (L1 sell / L2 buy)"

    stable_out_wei, swap_note = execute_l1_exact_input_swap(
        w3_l1_fork,
        token_in=weth_l1,
        token_out=stable_l1,
        fee_tier=l1_pool.fee_tier,
        amount_in_wei=amount_wei,
        recipient=swap_account,
    )
    if stable_out_wei is None:
        return None, swap_note

    buy_stable_wei = quoter_exact_output(
        w3_l2,
        "arbitrum",
        stable_l2,
        weth_l2,
        amount_wei,
        l2_pool.fee_tier,
        block_identifier=snapshot.l2_block,
    )
    if buy_stable_wei is None:
        return None, "quoter failed on L2 buy leg"

    sell_usd = stable_to_usd(stable_out_wei, stable_dec_l1)
    buy_usd = stable_to_usd(buy_stable_wei, stable_dec_l2)
    return sell_usd - buy_usd, f"{swap_note}; L2 buy via Quoter"


class AnvilProcess:
    """可选：本地 anvil fork 子进程。"""

    def __init__(
        self,
        fork_url: str,
        fork_block: int,
        *,
        port: int = 8545,
        startup_wait_sec: float = 5.0,
    ):
        self.fork_url = authenticated_fork_url(fork_url)
        self.fork_block = fork_block
        self.port = port
        self.startup_wait_sec = startup_wait_sec
        self._proc: subprocess.Popen | None = None
        self.rpc_url = f"http://127.0.0.1:{port}"

    @staticmethod
    def is_available() -> bool:
        return shutil.which("anvil") is not None

    def start(self) -> str | None:
        if not self.is_available():
            return None
        cmd = [
            "anvil",
            "--fork-url",
            self.fork_url,
            "--fork-block-number",
            str(self.fork_block),
            "--port",
            str(self.port),
            "--silent",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.time() + 45.0
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    break
                try:
                    w3 = create_sync_w3(self.rpc_url)
                    _ = w3.eth.block_number
                    return self.rpc_url
                except Exception:
                    time.sleep(0.5)
            if self._proc.poll() is not None:
                stderr = ""
                if self._proc.stderr is not None:
                    stderr = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                logger.warning(
                    "anvil exited early with code %s: %s",
                    self._proc.returncode,
                    stderr[:300] if stderr else "no stderr",
                )
                return None
            logger.warning("anvil RPC not ready within timeout on port %s", self.port)
            return None
        except Exception as exc:
            logger.warning("Failed to start anvil: %s", exc)
            self.stop()
            return None

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def verify_route(
    snapshot: MultiPoolSnapshot,
    route: CandidateRoute,
    amount_eth: float,
    *,
    w3_l1: Web3 | None = None,
    w3_l2: Web3 | None = None,
    use_anvil: bool = True,
    swap_exec: bool = False,
    model: MultiPoolCostModel | None = None,
) -> ForkVerifyResult:
    """对比模拟净利润 vs QuoterV2 / anvil fork（可选 L1 state-changing swap）。"""
    model = model or MultiPoolCostModel.from_routes(snapshot, [route])
    amount = min(max(amount_eth, 0.0), snapshot.inventory.max_amount_in_eth)
    simulated = model.evaluate_route_net_scalar(amount, route, snapshot)

    verification_mode = "rpc_quoter"
    anvil: AnvilProcess | None = None
    l1_rpc = RPC["ethereum"]["http"]
    l2_rpc = RPC["arbitrum"]["http"]

    if not l1_rpc or not l2_rpc:
        return ForkVerifyResult(
            snapshot_id=snapshot.snapshot_id,
            route_id=route.route_id,
            l1_pool_id=route.l1_pool_id,
            l2_pool_id=route.l2_pool_id,
            bridge_id=route.bridge_id,
            amount_in_eth=amount,
            l1_block=snapshot.l1_block,
            l2_block=snapshot.l2_block,
            simulated_profit=simulated,
            fork_profit=0.0,
            rel_error=1.0,
            verification_mode="unavailable",
            quoter_ok=False,
            notes="ETH_HTTP_URL / ARB_HTTP_URL not configured",
        )

    if swap_exec and not AnvilProcess.is_available():
        return ForkVerifyResult(
            snapshot_id=snapshot.snapshot_id,
            route_id=route.route_id,
            l1_pool_id=route.l1_pool_id,
            l2_pool_id=route.l2_pool_id,
            bridge_id=route.bridge_id,
            amount_in_eth=amount,
            l1_block=snapshot.l1_block,
            l2_block=snapshot.l2_block,
            simulated_profit=simulated,
            fork_profit=0.0,
            rel_error=1.0,
            verification_mode="unavailable",
            quoter_ok=False,
            notes="swap_exec requires Foundry anvil in PATH",
        )

    w3_l2 = w3_l2 or create_sync_w3(l2_rpc)
    w3_l1_local = w3_l1
    fork_rpc: str | None = None

    if (use_anvil or swap_exec) and AnvilProcess.is_available():
        anvil = AnvilProcess(l1_rpc, snapshot.l1_block)
        fork_rpc = anvil.start()
        if fork_rpc:
            w3_l1_local = create_sync_w3(fork_rpc)
            verification_mode = "anvil_quoter"

    if w3_l1_local is None:
        w3_l1_local = create_sync_w3(l1_rpc)

    try:
        if swap_exec and fork_rpc:
            gross, note = quote_cross_layer_gross_with_l1_swap(
                w3_l1_local, w3_l2, snapshot, route, amount
            )
            verification_mode = "anvil_l1_swap_l2_quoter"
            if gross is None:
                gross, note = quote_cross_layer_gross_usd(
                    w3_l1_local, w3_l2, snapshot, route, amount
                )
                verification_mode = "anvil_quoter"
                note = f"swap exec failed, fallback quoter: {note}"
        else:
            gross, note = quote_cross_layer_gross_usd(
                w3_l1_local, w3_l2, snapshot, route, amount
            )
            if fork_rpc and not swap_exec:
                verification_mode = "anvil_quoter"
        if gross is None:
            return ForkVerifyResult(
                snapshot_id=snapshot.snapshot_id,
                route_id=route.route_id,
                l1_pool_id=route.l1_pool_id,
                l2_pool_id=route.l2_pool_id,
                bridge_id=route.bridge_id,
                amount_in_eth=amount,
                l1_block=snapshot.l1_block,
                l2_block=snapshot.l2_block,
                simulated_profit=simulated,
                fork_profit=0.0,
                rel_error=1.0,
                verification_mode=verification_mode,
                quoter_ok=False,
                notes=note,
            )

        fork_profit = fork_net_profit_from_gross(gross, snapshot, route, amount, model)
        return ForkVerifyResult(
            snapshot_id=snapshot.snapshot_id,
            route_id=route.route_id,
            l1_pool_id=route.l1_pool_id,
            l2_pool_id=route.l2_pool_id,
            bridge_id=route.bridge_id,
            amount_in_eth=amount,
            l1_block=snapshot.l1_block,
            l2_block=snapshot.l2_block,
            simulated_profit=simulated,
            fork_profit=fork_profit,
            rel_error=rel_error(simulated, fork_profit),
            verification_mode=verification_mode,
            quoter_ok=True,
            notes=note,
        )
    finally:
        if anvil is not None:
            anvil.stop()


def append_fork_verify_result(
    result: ForkVerifyResult,
    path: Path | None = None,
) -> None:
    log_path = path or FORK_VERIFY_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
