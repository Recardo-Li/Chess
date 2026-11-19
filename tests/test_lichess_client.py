import json
import unittest
from unittest.mock import Mock

import requests

from chess_eval.lichess_client import (
    LichessClient,
    LichessError,
    LichessHTTPError,
    LichessNetworkError,
)


class LichessClientTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.session.headers = {}
        self.config = {
            "api_token": "private-token",
            "level": 3,
            "color": "black",
            "clock_limit": 600,
            "clock_increment": 10,
            "days": None,
        }
        self.client = LichessClient(self.config, session=self.session)

    def response(self, payload=None, *, status=200, lines=()):
        response = Mock()
        response.status_code = status
        response.headers = {}
        response.json.return_value = payload or {}
        response.iter_lines.return_value = iter(lines)
        self.session.request.return_value = response
        return response

    def test_ai_challenge_uses_form_data_and_bot_move_endpoint(self):
        self.response({"id": "abcdefgh", "player": "black"})
        self.assertEqual(self.client.create_ai_game()["id"], "abcdefgh")
        args, kwargs = self.session.request.call_args
        self.assertEqual(args, ("POST", "https://lichess.org/api/challenge/ai"))
        self.assertEqual(
            kwargs["data"],
            {"level": 3, "color": "black", "variant": "standard", "clock.limit": 600, "clock.increment": 10},
        )
        self.assertFalse(kwargs["allow_redirects"])
        self.client.make_move("abcdefgh", "a7a8q")
        self.assertEqual(
            self.session.request.call_args.args[1],
            "https://lichess.org/api/bot/game/abcdefgh/move/a7a8q",
        )

    def test_correspondence_omits_clock(self):
        self.client.config["clock_limit"] = None
        self.client.config["days"] = 2
        self.response({"id": "abcdefgh"})
        self.client.create_ai_game()
        data = self.session.request.call_args.kwargs["data"]
        self.assertEqual(data["days"], 2)
        self.assertNotIn("clock.limit", data)
        self.assertNotIn("clock.increment", data)

    def test_clock_takes_precedence_over_days(self):
        self.client.config["days"] = 1
        self.response({"id": "abcdefgh"})
        self.client.create_ai_game()
        data = self.session.request.call_args.kwargs["data"]
        self.assertEqual(data["clock.limit"], 600)
        self.assertEqual(data["clock.increment"], 10)
        self.assertNotIn("days", data)

    def test_non_bot_is_rejected_without_mutation(self):
        self.response({"id": "human"})
        with self.assertRaises(LichessError):
            self.client.require_bot_account()
        self.assertEqual(self.session.request.call_count, 1)
        self.assertEqual(self.session.request.call_args.args[0], "GET")

    def test_stream_skips_heartbeats_and_closes_after_terminal_state(self):
        events = [
            {"type": "gameFull", "state": {"moves": "", "status": "started"}},
            {"type": "gameState", "moves": "f2f3 e7e5 g2g4 d8h4", "status": "mate", "winner": "black"},
        ]
        response = self.response(lines=[b"", b" ", *(json.dumps(e).encode() for e in events)])
        self.assertEqual(list(self.client.stream_game("abcdefgh")), events)
        response.close.assert_called_once()

    def test_eof_before_game_end_requires_reconnect(self):
        response = self.response(lines=[b'{"type":"gameFull","state":{"status":"started"}}'])
        with self.assertRaises(LichessNetworkError):
            list(self.client.stream_game("abcdefgh"))
        response.close.assert_called_once()

    def test_move_timeout_is_not_retried_or_leaked(self):
        self.session.request.side_effect = requests.Timeout("private-token leaked upstream")
        with self.assertRaises(LichessNetworkError) as raised:
            self.client.make_move("abcdefgh", "e2e4")
        self.assertNotIn("private-token", str(raised.exception))
        self.assertEqual(self.session.request.call_count, 1)

    def test_http_status_and_retry_after_preserved_without_body(self):
        response = self.response(status=429)
        response.headers["Retry-After"] = "120"
        response.text = "private-token"
        with self.assertRaises(LichessHTTPError) as raised:
            self.client.get_account()
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.retry_after, 120)
        self.assertNotIn("private-token", str(raised.exception))
        response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
