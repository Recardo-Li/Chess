"""Small synchronous client for Lichess AI challenges and the Bot API.

The caller owns reconnects: a failed move request may already have been applied,
so mutations must never be retried without first checking the game stream.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import quote

import requests

from .contracts import EnvironmentError


class LichessError(EnvironmentError):
    """An API or protocol failure with a message safe to put in a log."""


class LichessNetworkError(LichessError):
    """Request outcome is unknown; reconnect and reconcile before a retry."""


class LichessHTTPError(LichessError):
    def __init__(self, status_code: int, retry_after: float | None = None):
        self.status_code = status_code
        self.retry_after = retry_after
        hint = {
            401: "请检查 Conifg/Lichess.yaml 中的 api_token。",
            403: "请检查 BOT 账号和 token 的 bot:play / challenge:write 权限。",
            429: "触发 Lichess 限流，请至少等待一分钟后重试。",
        }.get(status_code, "请检查配置、游戏状态和 Lichess 服务状态。")
        super().__init__(f"Lichess HTTP {status_code}；{hint}")


class LichessClient:
    def __init__(
        self, config: Mapping[str, Any], *, session: requests.Session | None = None
    ):
        self.config = dict(config)
        self.base_url = str(config.get("base_url", "https://lichess.org")).rstrip("/")
        self.request_timeout = float(config.get("request_timeout", 30))
        self.stream_timeout = float(config.get("stream_timeout", 90))
        self.session = session if session is not None else requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config['api_token']}",
                "Accept": "application/json",
                "User-Agent": "llm-chess-eval/1.0",
            }
        )

    def close(self) -> None:
        self.session.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        timeout = kwargs.pop("timeout", self.request_timeout)
        try:
            response = self.session.request(
                method,
                self.base_url + path,
                timeout=timeout,
                allow_redirects=False,
                **kwargs,
            )
        except requests.RequestException:
            # Never include raw requests exceptions: URLs/proxy errors can contain
            # credentials, and POST outcomes can be uncertain after a timeout.
            raise LichessNetworkError("Lichess 网络请求失败或超时；请求结果可能尚未确认。") from None
        if not 200 <= response.status_code < 300:
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = None
            status_code = response.status_code
            response.close()
            raise LichessHTTPError(status_code, retry_after)
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, path, **kwargs)
        try:
            payload = response.json()
        except ValueError:
            raise LichessError("Lichess 返回了无效 JSON。") from None
        finally:
            response.close()
        if not isinstance(payload, dict):
            raise LichessError("Lichess 返回了非对象 JSON。")
        return payload

    def get_account(self) -> dict[str, Any]:
        return self._json("GET", "/api/account")

    def require_bot_account(self) -> dict[str, Any]:
        account = self.get_account()
        if account.get("title") != "BOT":
            raise LichessError(
                "此 token 不属于 BOT 账号。请按 README 使用专用的新账号完成 BOT 升级；"
                "本程序不会自动执行不可逆的账号升级。"
            )
        return account

    def create_ai_game(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "level": self.config["level"],
            "color": self.config.get("color", "white"),
            "variant": "standard",
        }
        if self.config.get("clock_limit") is not None:
            data["clock.limit"] = self.config["clock_limit"]
            data["clock.increment"] = self.config["clock_increment"]
        else:
            data["days"] = self.config["days"]
        result = self._json("POST", "/api/challenge/ai", data=data)
        if not isinstance(result.get("id"), str) or not result["id"]:
            raise LichessError("Lichess 创建对局响应缺少 game id；请检查账号中的当前对局。")
        return result

    def make_move(self, game_id: str, move: str) -> dict[str, Any]:
        return self._json(
            "POST", f"/api/bot/game/{quote(game_id, safe='')}/move/{quote(move, safe='')}"
        )

    def resign(self, game_id: str) -> dict[str, Any]:
        return self._json("POST", f"/api/bot/game/{quote(game_id, safe='')}/resign")

    def stream_game(self, game_id: str) -> Iterator[dict[str, Any]]:
        """Yield one connection's NDJSON events, skipping blank heartbeats.

        EOF before a terminal game state is a reconnectable network error.
        Close this generator when leaving a game early to release its socket.
        """
        response = self._request(
            "GET",
            f"/api/bot/game/stream/{quote(game_id, safe='')}",
            headers={"Accept": "application/x-ndjson"},
            stream=True,
            timeout=(self.request_timeout, self.stream_timeout),
        )
        try:
            for line in response.iter_lines(chunk_size=1):
                if not line or not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    raise LichessError("Lichess 对局流包含无效 JSON。") from None
                if not isinstance(event, dict):
                    raise LichessError("Lichess 对局流包含非对象 JSON。")
                yield event
                state = event.get("state", {}) if event.get("type") == "gameFull" else event
                if event.get("type") in {"gameFull", "gameState"}:
                    status = state.get("status")
                    if status is not None and status not in {"created", "started"}:
                        return
        except requests.RequestException:
            raise LichessNetworkError("Lichess 对局流连接中断或超时。") from None
        finally:
            response.close()
        raise LichessNetworkError("Lichess 对局流在对局结束前关闭，需要重新同步。")
