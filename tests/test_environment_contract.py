"""Run the same evaluation loop with a non-chess task and two different agents."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from chess_eval.contracts import Decision, EpisodeResult, Feedback, Observation, Turn
from chess_eval.llm_client import LLMClient
from chess_eval.runner import GameRunner


class TokenGame:
    """One token remains. The player can take it or pass."""

    def __init__(self):
        self.action = None

    def turns(self, log):
        yield Turn(Observation("last-token", "There is one token left. Choose take or pass."))
        won = self.action == "take"
        yield EpisodeResult("win" if won else "loss", self.action, int(won))

    def validate(self, state_id, action):
        if action not in ("take", "pass"):
            return Feedback(action, "unknown_action", "Choose take or pass.")
        return None

    def submit(self, state_id, action):
        assert state_id == "last-token" and self.action is None
        self.action = action

    def finish(self, output_dir, outcome, agent_name):
        return {"last_action": self.action}


class PassingAgent:
    def act(self, observation, feedback=()):
        return Decision("pass")


class EnvironmentContractTests(unittest.TestCase):
    def test_replace_environment_and_agent_independently(self):
        skill = "Win by taking the last token."
        session = Mock()
        responses = []
        for content in ("take-three", "take"):
            response = Mock(status_code=200)
            response.json.return_value = {"choices": [{"message": {"content": content}}]}
            responses.append(response)
        session.post.side_effect = responses
        model = LLMClient({"model": "test"}, "test-key", skill, session=session)
        self.addCleanup(model.close)

        for agent, action, score, calls in [(model, "take", 1, 2), (PassingAgent(), "pass", 0, 0)]:
            with self.subTest(action=action), TemporaryDirectory() as tmp:
                runner = GameRunner(TokenGame(), agent, output_dir=Path(tmp),
                                    agent_name="player", skill_text=skill, settings={},
                                    invalid_action_retries=1)
                result = runner.run()
                self.assertEqual((result["last_action"], result["score"]), (action, score))
                self.assertEqual(result["api_calls"], calls)
                self.assertEqual(result["agent_attempts"], 2 if calls else 1)
                record = json.loads((runner.output_dir / "moves.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(record["action"], action)
                if calls:
                    self.assertEqual(record["attempts"][0]["error"], "unknown_action")
                    messages = session.post.call_args.kwargs["json"]["messages"]
                    self.assertEqual(messages[-2:], [
                        {"role": "assistant", "content": "take-three"},
                        {"role": "user", "content": "Choose take or pass."},
                    ])
