"""Exercise the configured CLI, real agent and real environment with offline HTTP."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import chess
import requests
import yaml

import run


LLM_KEY = "fixture-only-model-key"
LICHESS_TOKEN = "fixture-only-bot-token"
SKILL = "Play standard chess and give one UCI move."


def write_config(root, provider, include=True):
    config_dir = root / "Conifg"
    config_dir.mkdir()
    lichess = {
        "base_url": "https://chess.invalid", "api_token": LICHESS_TOKEN, "level": 2,
        "color": "white", "days": 1, "reconnect_attempts": 0,
        "request_timeout": 30, "stream_timeout": 30,
    }
    llm = {
        "base_url": "https://model.invalid/beta", "provider": provider, "model": "fixture",
        "include_legal_moves": include, "constrain_legal_moves": True,
        "max_plies": 10, "skill_path": "rules.md", "output_dir": "outputs",
    }
    (config_dir / "Lichess.yaml").write_text(yaml.safe_dump(lichess), encoding="utf-8")
    (config_dir / "LLM.yaml").write_text(yaml.safe_dump(llm), encoding="utf-8")
    (root / "rules.md").write_text(SKILL, encoding="utf-8")


def response(body=None, events=()):
    result = Mock(status_code=200)
    result.json.return_value = body
    result.iter_lines.return_value = [json.dumps(event).encode() for event in events]
    return result


class CliEpisodeTests(unittest.TestCase):
    def test_strict_episode_through_cli_for_each_provider(self):
        events = [
            {"type": "gameFull", "white": {"id": "bot"}, "black": {"aiLevel": 2},
             "state": {"moves": "", "status": "started"}},
            {"type": "gameState", "moves": "e2e4 e7e5", "status": "resign", "winner": "white"},
        ]
        for provider in ["gpt", "deepseek"]:
            with self.subTest(provider=provider), TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_config(root, provider)
                arguments = '{"move":"e2e4"}'
                message = {"content": arguments} if provider == "gpt" else {"tool_calls": [{
                    "type": "function", "function": {"name": "choose_move", "arguments": arguments},
                }]}
                replies = [
                    response({"id": "bot", "title": "BOT"}), response({"id": "abc123"}),
                    response(events=events),
                    response({"choices": [{"message": message}], "usage": {"total_tokens": 42}}),
                    response({"ok": True}),
                ]
                console = io.StringIO()
                with (patch.object(run, "ROOT", root), patch("sys.argv", ["run.py", "--api-key", LLM_KEY]),
                      patch.object(requests.Session, "request", autospec=True, side_effect=replies) as http,
                      redirect_stdout(console), redirect_stderr(console)):
                    self.assertEqual(run.main(), 0, console.getvalue())
                self.assertEqual([urlsplit(call.args[2]).path for call in http.call_args_list], [
                    "/api/account", "/api/challenge/ai", "/api/bot/game/stream/abc123",
                    "/beta/chat/completions", "/api/bot/game/abc123/move/e2e4",
                ])
                llm_request = http.call_args_list[3].kwargs
                self.assertEqual(llm_request["headers"]["Authorization"], "Bearer " + LLM_KEY)
                payload = llm_request["json"]
                self.assertIn(SKILL, payload["messages"][0]["content"])
                if provider == "gpt":
                    strict = payload["response_format"]["json_schema"]
                    schema = strict["schema"]
                else:
                    strict = payload["tools"][0]["function"]
                    self.assertEqual(strict["name"], "choose_move")
                    schema = strict["parameters"]
                self.assertTrue(strict["strict"])
                self.assertEqual(schema["required"], ["move"])
                self.assertEqual(set(schema["properties"]["move"]["enum"]),
                                 {move.uci() for move in chess.Board().legal_moves})

                (episode,) = (root / "outputs").iterdir()
                files = {path.name: path.read_text(encoding="utf-8") for path in episode.iterdir()}
                self.assertEqual(set(files), {"summary.json", "moves.jsonl", "game.pgn", "skill.md"})
                self.assertEqual(files["skill.md"], SKILL)
                self.assertIn("1. e4 e5", files["game.pgn"])
                summary = json.loads(files["summary.json"])
                self.assertEqual((summary["outcome"], summary["api_calls"], summary["total_tokens"]),
                                 ("win", 1, 42))
                recorded = console.getvalue() + "".join(files.values())
                for secret in [LLM_KEY, LICHESS_TOKEN]:
                    self.assertNotIn(secret, recorded)

    def test_hidden_options_with_strict_output_rejected_during_config_check(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_config(root, "gpt", include=False)
            error = io.StringIO()
            with (patch.object(run, "ROOT", root), patch("sys.argv", ["run.py", "--check-config"]),
                  patch.object(requests.Session, "request") as http, redirect_stderr(error)):
                self.assertEqual(run.main(), 1)
            self.assertIn("include_legal_moves", error.getvalue())
            http.assert_not_called()
