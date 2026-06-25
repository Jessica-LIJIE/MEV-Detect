"""Uniswap V3 池注册表加载（Phase 0）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import POOL_REGISTRY_PATH

SUPPORTED_CHAINS = ("ethereum", "arbitrum")


@dataclass(frozen=True)
class PoolRegistryEntry:
    id: str
    name: str
    chain: str
    pool_address: str
    token0: str
    token1: str
    token0_address: str
    token1_address: str
    fee_tier: int
    token0_decimals: int
    token1_decimals: int
    weth_is_token0: bool
    is_primary: bool = False
    dex: str = "uniswap_v3"

    def checksum_address(self) -> str:
        from web3 import Web3

        return Web3.to_checksum_address(self.pool_address)


_registry_cache: dict[str, Any] | None = None


def _registry_path(path: Path | None = None) -> Path:
    return path or POOL_REGISTRY_PATH


def load_registry_raw(path: Path | None = None) -> dict[str, Any]:
    global _registry_cache
    resolved = _registry_path(path)
    if path is None and _registry_cache is not None:
        return _registry_cache
    with open(resolved, encoding="utf-8") as f:
        data = json.load(f)
    if path is None:
        _registry_cache = data
    return data


def _resolve_token_address(chain: str, symbol: str, raw: dict[str, Any]) -> str:
    tokens = raw.get("tokens", {}).get(chain, {})
    address = tokens.get(symbol)
    if not address:
        raise KeyError(f"Token {symbol} not defined for chain {chain}")
    return address


def _parse_pool_entry(chain: str, item: dict[str, Any], raw: dict[str, Any]) -> PoolRegistryEntry:
    token0 = item["token0"]
    token1 = item["token1"]
    return PoolRegistryEntry(
        id=item["id"],
        name=item["name"],
        chain=chain,
        pool_address=item["pool_address"],
        token0=token0,
        token1=token1,
        token0_address=_resolve_token_address(chain, token0, raw),
        token1_address=_resolve_token_address(chain, token1, raw),
        fee_tier=int(item["fee_tier"]),
        token0_decimals=int(item["token0_decimals"]),
        token1_decimals=int(item["token1_decimals"]),
        weth_is_token0=bool(item["weth_is_token0"]),
        is_primary=bool(item.get("is_primary", False)),
        dex=item.get("dex", "uniswap_v3"),
    )


def load_registry(chain: str | None = None, path: Path | None = None) -> list[PoolRegistryEntry]:
    """加载池注册表。指定 chain 时只返回该链池列表。"""
    raw = load_registry_raw(path)
    pools_section = raw.get("pools", {})

    if chain is not None:
        if chain not in SUPPORTED_CHAINS:
            raise ValueError(f"Unsupported chain: {chain}. Expected one of {SUPPORTED_CHAINS}")
        items = pools_section.get(chain, [])
        return [_parse_pool_entry(chain, item, raw) for item in items]

    entries: list[PoolRegistryEntry] = []
    for chain_name in SUPPORTED_CHAINS:
        for item in pools_section.get(chain_name, []):
            entries.append(_parse_pool_entry(chain_name, item, raw))
    return entries


def get_primary_pool(chain: str, path: Path | None = None) -> PoolRegistryEntry:
    """返回 is_primary=true 的主池（tick 全量加载优先）。"""
    pools = load_registry(chain, path)
    for pool in pools:
        if pool.is_primary:
            return pool
    if not pools:
        raise ValueError(f"No pools registered for chain {chain}")
    return pools[0]


def get_contract_addresses(chain: str, path: Path | None = None) -> dict[str, str]:
    """返回 QuoterV2、Multicall3 等辅助合约地址。"""
    raw = load_registry_raw(path)
    return {
        "quoter_v2": raw.get("quoter_v2", {}).get(chain, ""),
        "multicall3": raw.get("multicall3", {}).get(chain, ""),
        "swap_router02": raw.get("swap_router02", {}).get(chain, ""),
    }


def registry_pool_count(path: Path | None = None) -> dict[str, int]:
    """各链池数量（用于自检）。"""
    raw = load_registry_raw(path)
    pools_section = raw.get("pools", {})
    return {chain: len(pools_section.get(chain, [])) for chain in SUPPORTED_CHAINS}
