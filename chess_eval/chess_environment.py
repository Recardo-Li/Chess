"""Standard chess observations, rules and authoritative Lichess game state."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import time

import chess
import chess.pgn

from .contracts import ActionSpec, EpisodeResult, Feedback, Observation, Turn
from .lichess_client import LichessClient, LichessError, LichessHTTPError, LichessNetworkError


_CHESS_ACTION = ActionSpec(field="move", tool="choose_move", text_mode="uci")
_UCI = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")


def chess_observation(
    board: chess.Board, state_id: str, *, include_legal_moves: bool, constrain_legal_moves: bool
) -> Observation:
    """Copy only the position and action constraints the agent is allowed to see."""
    side = "White" if board.turn == chess.WHITE else "Black"
    history = " ".join(move.uci() for move in board.move_stack) or "(none)"
    diagram = "\n".join(f"{8 - i}  {row}" for i, row in enumerate(str(board).splitlines()))
    prompt = (
        f"You play {side}; it is your turn.\n"
        f"Current FEN: {board.fen(en_passant='fen')}\n"
        f"Initial FEN: {board.root().fen(en_passant='fen')}\n"
        f"UCI move history from that initial position: {history}\n"
        f"Board (uppercase = White, lowercase = Black):\n{diagram}\n"
        "   a b c d e f g h\n"
        f"Your king is in check: {'yes' if board.is_check() else 'no'}.\n"
        "Select one legal move in lowercase UCI notation.\n"
    )
    choices = None
    if include_legal_moves:
        moves = tuple(sorted(move.uci() for move in board.legal_moves))
        prompt += "Legal moves: " + " ".join(moves) + "\n"
        if constrain_legal_moves:
            choices = moves
    return Observation(state_id=state_id, prompt=prompt, spec=_CHESS_ACTION, choices=choices)


@dataclass
class _PendingMove:
    ply: int
    uci: str
    uncertain: bool = False


class ChessEnvironment:
    def __init__(
        self, lichess: LichessClient, lichess_config: dict, *,
        include_legal_moves: bool, constrain_legal_moves: bool, max_plies: int,
    ):
        if constrain_legal_moves and not include_legal_moves:
            raise ValueError("启用 constrain_legal_moves 时必须同时设置 include_legal_moves: true。")
        self.lichess = lichess
        self.config = lichess_config
        self.include_legal_moves = include_legal_moves
        self.constrain_legal_moves = constrain_legal_moves
        self.max_plies = max_plies
        self.board = chess.Board()
        self.game_id: str | None = None
        self.color: str | None = None
        self.initial_fen = chess.STARTING_FEN
        self.state: dict = {}
        self._pending: _PendingMove | None = None
        self._active_state: str | None = None

    @property
    def game_url(self) -> str | None:
        if self.game_id is None:
            return None
        return f"{self.config['base_url'].rstrip('/')}/{self.game_id}"

    def _identify(self, event: dict, account_id: str) -> None:
        if event.get("variant", {}).get("key", "standard") != "standard":
            raise LichessError("收到非标准国际象棋对局，停止评测。")
        for side, other in (("white", "black"), ("black", "white")):
            if str(event.get(side, {}).get("id", "")).lower() == account_id.lower():
                if not event.get(other, {}).get("aiLevel"):
                    raise LichessError("对手不是 Lichess 内置 AI，停止评测。")
                fen = event.get("initialFen", "startpos")
                initial_fen = chess.STARTING_FEN if fen == "startpos" else fen
                if self.color is not None and (self.color != side or self.initial_fen != initial_fen):
                    raise LichessError("重新同步的执棋方或初始局面与本局不一致。")
                self.color, self.initial_fen = side, initial_fen
                return
        raise LichessError("对局玩家与 BOT 账号不匹配。")

    def _sync(self, state: dict, log: Callable[[dict], None]) -> None:
        try:
            board = chess.Board(self.initial_fen)
            for uci in state.get("moves", "").split():
                board.push_uci(uci)
        except (ValueError, TypeError, AttributeError):
            raise LichessError("Lichess 返回的棋局状态无法还原。") from None
        previous_plies = len(self.board.move_stack)
        if board.move_stack[:previous_plies] != self.board.move_stack:
            raise LichessError("服务端棋谱发生回退或分歧，停止评测。")
        self.board, self.state = board, dict(state)

        pending = self._pending
        if pending and len(board.move_stack) > pending.ply:
            if board.move_stack[pending.ply].uci() != pending.uci:
                raise LichessError("服务端落子与模型提交不一致。")
            log({"type": "move_confirmed", "ply": pending.ply, "uci": pending.uci})
            self._pending = None
        colors = ("white", "black") if board.root().turn else ("black", "white")
        for ply in range(previous_plies, len(board.move_stack)):
            print(f"{ply + 1:03d} {colors[ply % 2]}: {board.move_stack[ply].uci()}", flush=True)

    def _terminal_result(self) -> EpisodeResult | None:
        status = self.state.get("status")
        if status in (None, "created", "started"):
            return None
        winner = self.state.get("winner")
        if winner in ("white", "black"):
            won = winner == self.color
            return EpisodeResult(
                outcome="win" if won else "loss", result="1-0" if winner == "white" else "0-1",
                score=1 if won else 0,
            )
        if status in ("draw", "stalemate", "outoftime", "timeout", "insufficientMaterialClaim"):
            return EpisodeResult(outcome="draw", result="1/2-1/2", score=0.5)
        return EpisodeResult(outcome="incomplete")

    def turns(self, log: Callable[[dict], None]) -> Generator[Turn | EpisodeResult, None, None]:
        account = self.lichess.require_bot_account()
        self.game_id = self.lichess.create_ai_game()["id"]
        print(f"对局：{self.game_url}", flush=True)
        budget = self.config["reconnect_attempts"]
        for connection in range(budget + 1):
            try:
                with closing(self.lichess.stream_game(self.game_id)) as events:
                    for event in events:
                        kind = event.get("type")
                        if kind == "gameFull":
                            self._identify(event, account["id"])
                            state = event["state"]
                        elif kind == "gameState":
                            if self.color is None:
                                raise LichessError("首次同步缺少 gameFull。")
                            state = event
                        else:
                            continue
                        self._sync(state, log)
                        result = self._terminal_result()
                        if result is not None:
                            yield result
                            return
                        if self._pending:
                            if kind == "gameFull" and self._pending.uncertain:
                                raise LichessError("落子请求未获确认，重新同步后仍无该走法；停止并保留记录。")
                            continue
                        ply = len(self.board.move_stack)
                        if ply >= self.max_plies:
                            yield EpisodeResult(outcome="ply_limit", reason="达到 max_plies；不计为和棋。")
                            return
                        if self.board.is_game_over() or self.board.turn != (self.color == "white"):
                            continue
                        state_id = f"{self.game_id}/{ply}"
                        observation = chess_observation(
                            self.board, state_id, include_legal_moves=self.include_legal_moves,
                            constrain_legal_moves=self.constrain_legal_moves,
                        )
                        self._active_state = state_id
                        try:
                            yield Turn(
                                observation, log_fields={"ply": ply, "fen": self.board.fen()},
                                action_key="uci", invalid_outcome="invalid_move",
                            )
                        finally:
                            self._active_state = None
                        if self._pending is None:
                            raise LichessError("本回合尚未提交走法。")
                        if self._pending.uncertain:
                            # Interrupt this connection and reconcile against a fresh snapshot.
                            raise LichessNetworkError("落子响应不确定，重新读取服务端状态。")
                raise LichessNetworkError("对局流提前结束。")
            except LichessNetworkError:
                if connection == budget:
                    raise
                print(f"正在重新同步对局（{connection + 1}/{budget}）…", flush=True)
                time.sleep(min(connection + 1, 3))

    def validate(self, state_id: str, action: str) -> Feedback | None:
        if self._active_state is None or state_id != self._active_state:
            raise LichessError("该局面的决策已失效。")
        try:
            move = chess.Move.from_uci(action) if _UCI.fullmatch(action) else None
        except ValueError:
            move = None
        if move is None:
            return Feedback(action, "invalid_uci_format", "Return exactly one move in lowercase UCI notation.")
        if move not in self.board.legal_moves:
            return Feedback(action, "illegal_move", "The move is illegal in this position. Choose a legal UCI move.")
        return None

    def submit(self, state_id: str, action: str) -> None:
        if self.validate(state_id, action) is not None:
            raise LichessError("不能向 Lichess 提交非法走法。")
        self._active_state = None
        self._pending = _PendingMove(len(self.board.move_stack), action)
        try:
            self.lichess.make_move(self.game_id, action)
        except LichessHTTPError as exc:
            if exc.status_code not in (400, 409, 500, 502, 503, 504):
                raise
            self._pending.uncertain = True
        except LichessNetworkError:
            self._pending.uncertain = True

    def finish(self, output_dir: Path, outcome: EpisodeResult, agent_name: str) -> dict:
        details = {
            "game_id": self.game_id, "game_url": self.game_url, "llm_color": self.color,
            "lichess_status": self.state.get("status"), "confirmed_plies": len(self.board.move_stack),
            "final_fen": self.board.fen(),
        }
        if self.game_id and self.state.get("status", "started") in ("created", "started"):
            try:
                self.lichess.resign(self.game_id)
                details["cleanup"] = "resign_requested"
            except LichessError as exc:
                details["cleanup"], details["cleanup_error"] = "resign_failed", str(exc)
                print("认输请求未确认，请打开对局链接检查。", flush=True)
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update({
            "Event": "LLM vs Lichess AI", "Site": self.game_url or "?",
            "Date": datetime.now(timezone.utc).strftime("%Y.%m.%d"),
            "Result": outcome.result, "Termination": outcome.outcome,
        })
        for side in ("white", "black"):
            game.headers[side.title()] = agent_name if side == self.color else f"Lichess AI level {self.config['level']}"
        (output_dir / "game.pgn").write_text(str(game) + "\n", encoding="utf-8")
        return details
