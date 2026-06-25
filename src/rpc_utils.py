"""Infura RPC 连接辅助（支持 API Key Secret 认证）。

HTTP 请求使用 requests（同步），避免 Windows 上 aiohttp 连接超时。
WebSocket 仍用于实时订阅。
"""

import base64
import os
from typing import TYPE_CHECKING

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from web3 import Web3
from web3.providers.rpc import HTTPProvider

if TYPE_CHECKING:
    from web3 import Web3 as Web3Type

load_dotenv()

INFURA_API_KEY_SECRET = os.getenv("INFURA_API_KEY_SECRET", "")


def authenticated_fork_url(http_url: str, api_secret: str | None = None) -> str:
    """为 anvil --fork-url 嵌入 Infura Basic 认证（空用户名 + Secret）。

    web3 的 Session.auth 无法用于 anvil 子进程，需在 URL 中携带 Secret。
    """
    from urllib.parse import urlparse, urlunparse

    secret = api_secret if api_secret is not None else INFURA_API_KEY_SECRET
    if not http_url or not secret or "infura.io" not in http_url.lower():
        return http_url

    parsed = urlparse(http_url)
    if parsed.username is not None or parsed.password is not None:
        return http_url

    host = parsed.hostname or ""
    port_suffix = f":{parsed.port}" if parsed.port else ""
    netloc = f":{secret}@{host}{port_suffix}"
    return urlunparse(
        (parsed.scheme, netloc, parsed.path or "", parsed.params, parsed.query, parsed.fragment)
    )


def _auth_headers() -> dict[str, str]:
    if not INFURA_API_KEY_SECRET:
        return {}
    token = base64.b64encode(f":{INFURA_API_KEY_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def get_requests_auth() -> tuple[str, str] | None:
    if not INFURA_API_KEY_SECRET:
        return None
    return ("", INFURA_API_KEY_SECRET)


def create_sync_w3(http_url: str) -> "Web3Type":
    """创建基于 requests 的同步 Web3 客户端（Windows 上更稳定）。"""
    session = requests.Session()
    auth = get_requests_auth()
    if auth:
        session.auth = auth

    retry = Retry(
        total=5,
        connect=5,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return Web3(HTTPProvider(http_url, session=session, request_kwargs={"timeout": 30}))


def get_websocket_kwargs() -> dict:
    headers = _auth_headers()
    if not headers:
        return {}
    return {"extra_headers": headers}


def get_http_request_kwargs() -> dict:
    """保留给少数仍使用 AsyncHTTPProvider 的场景。"""
    headers = _auth_headers()
    if not headers:
        return {}
    return {"headers": headers}
