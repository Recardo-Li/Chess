"""Behavioral loop tests using server snapshots and no external requests."""

import json
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess

from chess_eval.chess_environment import ChessEnvironment
from chess_eval.contracts import AgentOutputError, Decision, EnvironmentError, Observation
from chess_eval.lichess_client import LichessHTTPError, LichessNetworkError
from chess_eval.runner import GameRunner


def metadata(raw="e2e4", error=None):
    return {
        "api_calls": 1, "elapsed_seconds": 0.25, "raw_response": raw, "error": error,
        "usage": {"total_tokens": 30},
    }


def state(moves="", status="started", winner=None):
    value = {"type": "gameState", "moves": moves, "status": status}
    if winner is not None:
        value["winner"] = winner
    return value


def full(color="white", moves="", status="started", winner=None):
    value = {
        "type": "gameFull", "id": "game1234", "initialFen": "startpos",
        "variant": {"key": "standard"},
        "white": {"id": "testbot"} if color == "white" else {"aiLevel": 1},
        "black": {"id": "testbot"} if color == "black" else {"aiLevel": 1},
        "state": state(moves, status, winner),
    }
    return value


class FakeLichess:
    def __init__(self, connections, move_errors=None):
        self.connections = list(connections)
        self.move_errors = list(move_errors or [])
        self.moves = []
        self.resignations = []
        self.stream_calls = 0
        self.closed_streams = 0

    def require_bot_account(self):
        return {"id": "testbot", "title": "BOT"}

    def create_ai_game(self):
        return {"id": "game1234"}

    def stream_game(self, game_id):
        if game_id != "game1234":
            raise AssertionError("wrong game id")
        script = self.connections[self.stream_calls]
        self.stream_calls += 1
        try:
            for event in script:
                if isinstance(event, Exception):
                    raise event
                yield event
        finally:
            self.closed_streams += 1

    def make_move(self, game_id, uci):
        self.moves.append((game_id, uci))
        if self.move_errors:
            error = self.move_errors.pop(0)
            if error is not None:
                raise error
        return {"ok": True}

    def resign(self, game_id):
        self.resignations.append(game_id)
        return {"ok": True}


class FakeAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.observations = []
        self.feedback = []

    def act(self, observation, feedback=()):
        self.observations.append(observation)
        self.feedback.append(feedback)
        if not self.responses:
            raise AssertionError("LLM was called more often than expected")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return Decision(response, metadata(response))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.sleep_patch = patch("chess_eval.chess_environment.time.sleep")
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)
        self.print_patch = patch("builtins.print")
        self.print_mock = self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def runner(self, connections, responses, *, max_plies=200, reconnect_attempts=2, move_errors=None,
               include_legal_moves=False, constrain_legal_moves=False, retries=0):
        self.lichess = FakeLichess(connections, move_errors)
        self.agent = FakeAgent(responses)
        lc = {"base_url": "https://lichess.org", "level": 1, "reconnect_attempts": reconnect_attempts}
        self.environment = ChessEnvironment(
            self.lichess, {**lc, "api_token": "private-lichess-token"},
            include_legal_moves=include_legal_moves, constrain_legal_moves=constrain_legal_moves,
            max_plies=max_plies,
        )
        return GameRunner(
            self.environment, self.agent, output_dir=self.root / "outputs", agent_name="test-model",
            skill_text="Test chess skill", settings={"lichess": lc}, invalid_action_retries=retries,
        )

    def test_complete_games_white_and_black_wins_and_losses(self):
        games = [
            ("white", "white", "e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7"),
            ("white", "black", "f2f3 e7e5 g2g4 d8h4"),
            ("black", "black", "f2f3 e7e5 g2g4 d8h4"),
            ("black", "white", "e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7"),
        ]
        for color, winner, line in games:
            with self.subTest(color=color, winner=winner):
                moves = line.split()
                script = [full(color)]
                for count in range(1, len(moves) + 1):
                    terminal = count == len(moves)
                    script.append(state(" ".join(moves[:count]), "mate" if terminal else "started",
                                        winner if terminal else None))
                expected_moves = moves[0 if color == "white" else 1::2]
                runner = self.runner([script], expected_moves)
                summary = runner.run()
                self.assertEqual(summary["outcome"], "win" if color == winner else "loss")
                self.assertEqual(summary["score"], 1 if color == winner else 0)
                self.assertEqual(summary["result"], "1-0" if winner == "white" else "0-1")
                self.assertEqual(summary["confirmed_plies"], len(moves))
                self.assertEqual(self.lichess.moves, [("game1234", move) for move in expected_moves])
                self.assertEqual(self.lichess.resignations, [])
                self.assertEqual(self.lichess.closed_streams, 1)
                pgn = (runner.output_dir / "game.pgn").read_text(encoding="utf-8")
                self.assertIn(f'[Result "{summary["result"]}"]', pgn)
                saved = json.loads((runner.output_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["api_calls"], len(expected_moves))
                self.assertNotIn("private-lichess-token", json.dumps(saved))

    def test_duplicate_states_do_not_repeat_llm_or_move_post(self):
        script = [full(), state(), full(), state("e2e4"), state("e2e4"),
                  state("e2e4 e7e5"), state("e2e4 e7e5"),
                  state("e2e4 e7e5 g1f3", "resign", "white")]
        runner = self.runner([script], ["e2e4", "g1f3"])
        self.assertEqual(runner.run()["outcome"], "win")
        self.assertEqual(len(self.agent.observations), 2)
        self.assertEqual(len(self.lichess.moves), 2)
        move_lines = [call.args[0] for call in self.print_mock.call_args_list
                      if call.args and call.args[0][:3].isdigit()]
        self.assertEqual(move_lines, ["001 white: e2e4", "002 black: e7e5", "003 white: g1f3"])

    def test_illegal_model_move_is_failure_and_requests_resignation(self):
        runner = self.runner([[full()]], ["e2e5"])
        summary = runner.run()
        self.assertEqual(summary["outcome"], "invalid_move")
        self.assertEqual(summary["score"], 0)
        self.assertEqual(summary["invalid_responses"], 1)
        self.assertEqual(summary["legal_response_rate"], 0)
        self.assertEqual(self.lichess.moves, [])
        self.assertEqual(self.lichess.resignations, ["game1234"])
        self.assertEqual(summary["cleanup"], "resign_requested")
        event = json.loads((runner.output_dir / "moves.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(event["attempts"][0]["raw_response"], "e2e5")

    def test_ply_limit_is_not_draw_and_does_not_call_llm(self):
        runner = self.runner([[full(moves="e2e4 e7e5")]], [], max_plies=2)
        summary = runner.run()
        self.assertEqual(summary["outcome"], "ply_limit")
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["result"], "*")
        self.assertEqual(self.agent.observations, [])
        self.assertEqual(self.lichess.resignations, ["game1234"])

    def test_output_error_is_saved_without_scoring_illegal_move_or_submitting(self):
        failure = AgentOutputError("Truncated response", metadata("", "output_truncated"))
        runner = self.runner([[full()]], [failure], retries=2)
        summary = runner.run()
        self.assertEqual(summary["outcome"], "output_error")
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["output_error_responses"], 1)
        self.assertEqual(summary["invalid_responses"], 0)
        self.assertEqual(self.lichess.moves, [])
        self.assertEqual(self.lichess.resignations, ["game1234"])

    def test_same_observation_is_retried_with_feedback_and_all_attempts_logged(self):
        runner = self.runner([[full(), state("e2e4", "resign", "white")]],
                             ["e2e5", "e2e4"], retries=1)
        summary = runner.run()
        self.assertEqual(summary["outcome"], "win")
        self.assertEqual(self.lichess.moves, [("game1234", "e2e4")])
        first, second = self.agent.observations
        self.assertIsInstance(first, Observation)
        self.assertEqual(first, second)
        self.assertFalse(self.agent.feedback[0])
        self.assertEqual([(f.action, f.code) for f in self.agent.feedback[1]],
                         [("e2e5", "illegal_move")])
        events = (runner.output_dir / "moves.jsonl").read_text(encoding="utf-8").splitlines()
        decision = next(event for event in map(json.loads, events) if event["type"] == "decision")
        self.assertEqual(decision["uci"], "e2e4")
        self.assertEqual([(a["attempt"], a["valid"]) for a in decision["attempts"]],
                         [(1, False), (2, True)])
        self.assertEqual(summary["legal_response_rate"], 0.5)
        self.assertEqual(summary["total_tokens"], 60)

    def test_environment_classifies_non_uci_and_illegal_actions(self):
        invalid_format = ["I play e2e4", '{"move":"e2e4"}', "```e2e4```", "E2E4", "0000", "e2e2"]
        for action in [*invalid_format, "e2e5", "e1g1"]:
            with self.subTest(action=action):
                runner = self.runner([[full()]], [action])
                summary = runner.run()
                self.assertEqual(summary["outcome"], "invalid_move")
                self.assertEqual(self.lichess.moves, [])
                self.assertEqual(summary["final_fen"], chess.STARTING_FEN)
                log = json.loads((runner.output_dir / "moves.jsonl").read_text(encoding="utf-8"))
                expected = "invalid_uci_format" if action in invalid_format else "illegal_move"
                self.assertEqual(log["attempts"][0]["error"], expected)

    def test_observation_and_all_three_candidate_visibility_modes(self):
        history = "e2e4 d7d5 f1b5"
        board = chess.Board()
        for uci in history.split():
            board.push_uci(uci)
        legal = sorted(move.uci() for move in board.legal_moves)
        for include, strict in [(False, False), (True, False), (True, True)]:
            with self.subTest(include=include, strict=strict):
                runner = self.runner([[full("black", history),
                                       state(history + " c7c6", "resign", "black")]], ["c7c6"],
                                     include_legal_moves=include, constrain_legal_moves=strict)
                self.assertEqual(runner.run()["outcome"], "win")
                (observation,) = self.agent.observations
                self.assertIn("You play Black", observation.prompt)
                self.assertIn(board.root().fen(), observation.prompt)
                self.assertIn(board.fen(en_passant="fen"), observation.prompt)
                self.assertIn(history, observation.prompt)
                self.assertIn("Your king is in check: yes", observation.prompt)
                self.assertIn("a b c d e f g h", observation.prompt)
                for index, row in enumerate(str(board).splitlines()):
                    self.assertIn(f"{8 - index}  {row}", observation.prompt)
                if include:
                    line = next(line for line in observation.prompt.splitlines()
                                if line.startswith("Legal moves:"))
                    self.assertEqual(sorted(line.removeprefix("Legal moves:").split()), legal)
                else:
                    self.assertNotIn("Legal moves:", observation.prompt)
                    self.assertNotIn("c7c6", observation.prompt)
                self.assertEqual(observation.choices, tuple(legal) if strict else None)

    def test_castling_en_passant_and_promotion_are_legal_actions(self):
        examples = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("8/P7/8/8/8/8/7k/4K3 w - - 0 1", "a7a8n"),
        ]
        for fen, action in examples:
            with self.subTest(action=action):
                start = {**full(), "initialFen": fen}
                runner = self.runner([[start, state(action, "resign", "white")]], [action],
                                     include_legal_moves=True, constrain_legal_moves=True)
                self.assertEqual(runner.run()["outcome"], "win")
                (observation,) = self.agent.observations
                self.assertIn(action, observation.choices)
                self.assertEqual(self.lichess.moves, [("game1234", action)])

    def test_local_checkmate_waits_for_server_result_without_requesting_action(self):
        mate = "f2f3 e7e5 g2g4 d8h4"
        runner = self.runner([[full(moves=mate), state(mate, "mate", "black")]], [])
        summary = runner.run()
        self.assertEqual(summary["result"], "0-1")
        self.assertEqual(self.agent.observations, [])
        self.assertEqual(self.lichess.moves, [])

    def test_old_state_id_cannot_submit_on_a_new_turn(self):
        self.runner([[full(), state("e2e4 e7e5")]], [])
        with closing(self.environment.turns(lambda event: None)) as turns:
            first = next(turns).observation
            self.assertIsNone(self.environment.validate(first.state_id, "e2e4"))
            self.environment.submit(first.state_id, "e2e4")
            second = next(turns).observation
            self.assertNotEqual(first.state_id, second.state_id)
            self.assertIsNone(self.environment.validate(second.state_id, "g1f3"))
            with self.assertRaises(EnvironmentError):
                self.environment.submit(first.state_id, "g1f3")
            self.assertEqual(self.lichess.moves, [("game1234", "e2e4")])

    def test_server_terminal_result_takes_priority_over_ply_limit(self):
        runner = self.runner([[full(moves="f2f3 e7e5 g2g4 d8h4", status="mate", winner="black")]],
                             [], max_plies=1)
        summary = runner.run()
        self.assertEqual(summary["outcome"], "loss")
        self.assertEqual(summary["score"], 0)
        self.assertEqual(self.agent.observations, [])
        self.assertEqual(self.lichess.resignations, [])

    def test_server_draw_is_recorded_without_inventing_local_result(self):
        for status in ("draw", "stalemate", "outoftime", "timeout", "insufficientMaterialClaim"):
            with self.subTest(status=status):
                runner = self.runner([[full(status=status)]], [])
                summary = runner.run()
                self.assertEqual(summary["outcome"], "draw")
                self.assertEqual(summary["score"], 0.5)
                self.assertEqual(summary["result"], "1/2-1/2")
                self.assertEqual(self.agent.observations, [])

    def test_aborted_game_is_not_a_draw(self):
        runner = self.runner([[full(status="aborted")]], [])
        summary = runner.run()
        self.assertEqual(summary["outcome"], "incomplete")
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["result"], "*")

    def test_stream_disconnect_reconnects_and_preserves_confirmed_history(self):
        runner = self.runner([
            [full(), state("e2e4"), LichessNetworkError("disconnected")],
            [full(moves="e2e4"), state("e2e4 e7e5"), state("e2e4 e7e5 g1f3", "resign", "white")],
        ], ["e2e4", "g1f3"])
        self.assertEqual(runner.run()["outcome"], "win")
        self.assertEqual(self.lichess.stream_calls, 2)
        self.assertEqual(self.lichess.closed_streams, 2)
        self.assertEqual(len(self.agent.observations), 2)

    def test_stream_eof_reconnects_without_repeating_pending_move(self):
        runner = self.runner([
            [full()],
            [full(moves="e2e4"), state("e2e4 e7e5", "resign", "white")],
        ], ["e2e4"])
        self.assertEqual(runner.run()["outcome"], "win")
        self.assertEqual(len(self.agent.observations), 1)
        self.assertEqual(len(self.lichess.moves), 1)

    def test_uncertain_post_accepted_on_server_is_confirmed_without_resubmission(self):
        for error in (LichessNetworkError("timeout"), LichessHTTPError(500), LichessHTTPError(400)):
            with self.subTest(error=type(error).__name__):
                runner = self.runner([
                    [full()],
                    [full(moves="e2e4"), state("e2e4 e7e5", "resign", "white")],
                ], ["e2e4"], move_errors=[error])
                summary = runner.run()
                self.assertEqual(summary["outcome"], "win")
                self.assertEqual(len(self.agent.observations), 1)
                self.assertEqual(len(self.lichess.moves), 1)
                self.assertEqual(summary["confirmed_plies"], 2)

    def test_uncertain_post_missing_after_snapshot_stops_without_resubmission(self):
        runner = self.runner([[full()], [full()]], ["e2e4"],
                             move_errors=[LichessNetworkError("timeout")])
        summary = runner.run()
        self.assertEqual(summary["outcome"], "error")
        self.assertEqual(len(self.agent.observations), 1)
        self.assertEqual(len(self.lichess.moves), 1)
        self.assertEqual(self.lichess.stream_calls, 2)
        self.assertEqual(self.lichess.resignations, ["game1234"])

    def test_terminal_snapshot_after_uncertain_post_takes_priority(self):
        runner = self.runner([[full()], [full(status="outoftime", winner="black")]],
                             ["e2e4"], move_errors=[LichessNetworkError("timeout")])
        summary = runner.run()
        self.assertEqual(summary["outcome"], "loss")
        self.assertEqual(summary["result"], "0-1")
        self.assertEqual(self.lichess.resignations, [])

    def test_reconnect_budget_exhaustion_is_error_not_draw(self):
        runner = self.runner([
            [full("black"), LichessNetworkError("disconnect")],
            [full("black"), LichessNetworkError("disconnect")],
        ], [], reconnect_attempts=1)
        summary = runner.run()
        self.assertEqual(summary["outcome"], "error")
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["result"], "*")
        self.assertEqual(self.lichess.stream_calls, 2)

    def test_server_history_rollback_is_error(self):
        runner = self.runner([
            [full(), state("e2e4"), LichessNetworkError("disconnect")],
            [full()],
        ], ["e2e4"])
        self.assertEqual(runner.run()["outcome"], "error")
        self.assertEqual(len(self.agent.observations), 1)


if __name__ == "__main__":
    unittest.main()
