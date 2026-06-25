"""RPC 连通性诊断。用法: python scripts/test_rpc.py"""

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from config.settings import INFURA_API_KEY_SECRET, RPC
from src.rpc_utils import get_requests_auth


def test_url(name: str, url: str) -> bool:
    if not url:
        print(f"[FAIL] {name}: 未配置（.env 为空）")
        return False

    if "/v3/" in url:
        masked = url.split("/v3/")[0] + "/v3/" + url.split("/v3/")[1][:4] + "****"
    else:
        masked = url[:40] + "..."
    print(f"\n测试 {name}")
    print(f"  URL: {masked}")
    print(f"  API Secret: {'已配置' if INFURA_API_KEY_SECRET else '未配置'}")

    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    try:
        resp = requests.post(
            url,
            json=payload,
            auth=get_requests_auth(),
            timeout=20,
        )
    except requests.exceptions.ConnectTimeout:
        print("  [FAIL] 连接超时 — 可能是网络/防火墙问题，可尝试开 VPN")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"  [FAIL] 无法连接: {e}")
        return False

    print(f"  HTTP 状态码: {resp.status_code}")
    text = resp.text.strip()
    if resp.status_code == 200 and "result" in text:
        block_hex = resp.json().get("result", "")
        block_num = int(block_hex, 16) if block_hex else 0
        print(f"  [OK] 连通成功，当前区块: {block_num}")
        return True

    print(f"  [FAIL] 响应: {text[:120]}")
    lower = text.lower()
    if "invalid project" in lower:
        print("  提示: API Key 无效，请到 Infura 重新复制 API Key。")
    elif "private key only" in lower or "api key secret" in lower:
        print("  提示: Infura 要求 API Key Secret。请任选其一：")
        print("    方案 A: Infura 控制台 → API Key → Settings")
        print("           关闭「Require API Key Secret for all requests」")
        print("    方案 B: 在 .env 添加 INFURA_API_KEY_SECRET=你的Secret")
    return False


def main():
    print("=== Infura RPC 诊断 ===\n")
    ok_eth = test_url("ETH_HTTP_URL", RPC["ethereum"]["http"])
    ok_arb = test_url("ARB_HTTP_URL", RPC["arbitrum"]["http"])

    print()
    if ok_eth and ok_arb:
        print("全部通过！可运行:")
        print("  python -m pytest tests/test_listener.py::test_rpc_fetch_latest_snapshot -v")
        sys.exit(0)
    print("存在问题，请按上方提示修复 .env 或 Infura 设置后重试。")
    sys.exit(1)


if __name__ == "__main__":
    main()
