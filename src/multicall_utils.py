"""Multicall3 aggregate3 批量 eth_call（Phase 1）。"""

from __future__ import annotations

from typing import Any

from eth_abi import decode
from web3 import Web3
from web3.contract.contract import ContractFunction

MULTICALL3_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "target", "type": "address"},
                    {"name": "allowFailure", "type": "bool"},
                    {"name": "callData", "type": "bytes"},
                ],
                "name": "calls",
                "type": "tuple[]",
            }
        ],
        "name": "aggregate3",
        "outputs": [
            {
                "components": [
                    {"name": "success", "type": "bool"},
                    {"name": "returnData", "type": "bytes"},
                ],
                "name": "returnData",
                "type": "tuple[]",
            }
        ],
        "stateMutability": "payable",
        "type": "function",
    }
]


def encode_contract_call(fn: ContractFunction) -> bytes:
    return bytes.fromhex(fn._encode_transaction_data()[2:])


def aggregate3_call(
    w3: Web3,
    multicall_address: str,
    calls: list[tuple[str, bytes]],
    block_identifier: int | str,
    *,
    allow_failure: bool = False,
) -> list[tuple[bool, bytes]]:
    """执行 Multicall3.aggregate3，calls 为 (target, calldata) 列表。"""
    if not calls:
        return []

    mc = w3.eth.contract(
        address=Web3.to_checksum_address(multicall_address),
        abi=MULTICALL3_ABI,
    )
    payload = [
        (Web3.to_checksum_address(target), allow_failure, calldata)
        for target, calldata in calls
    ]
    raw = mc.functions.aggregate3(payload).call(block_identifier=block_identifier)
    return [(bool(item[0]), bytes(item[1])) for item in raw]


def decode_slot0(return_data: bytes) -> tuple[int, int]:
    sqrt_price_x96, tick, *_rest = decode(
        ["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
        return_data,
    )
    return int(sqrt_price_x96), int(tick)


def decode_liquidity(return_data: bytes) -> int:
    (liquidity,) = decode(["uint128"], return_data)
    return int(liquidity)


def decode_tick(return_data: bytes) -> dict[str, Any]:
    (
        liquidity_gross,
        liquidity_net,
        _fg0,
        _fg1,
        _tc,
        _spl,
        _sec,
        initialized,
    ) = decode(
        [
            "uint128",
            "int128",
            "uint256",
            "uint256",
            "int56",
            "uint160",
            "uint32",
            "bool",
        ],
        return_data,
    )
    return {
        "liquidityGross": str(int(liquidity_gross)),
        "liquidityNet": str(int(liquidity_net)),
        "initialized": bool(initialized),
    }
