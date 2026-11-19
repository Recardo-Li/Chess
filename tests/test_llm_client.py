import unittest
from unittest.mock import Mock

import requests

from chess_eval.contracts import ActionSpec, AgentError, AgentOutputError, Feedback, Observation
from chess_eval.llm_client import LLMClient


TEXT = Observation("p1", "You are at a fork in the road.")
CHOICES = Observation("p1", TEXT.prompt, choices=("left", "right"))


def response(content="left", status=200):
    result = Mock()
    result.status_code = status
    result.json.return_value = {
        "choices": [{"message": {"content": content, "reasoning_content": "PRIVATE REASONING"}}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 5, "total_tokens": 85},
    }
    return result


def tool_response(arguments='{"action":"left"}', name="choose_action", finish="tool_calls"):
    result = response(None)
    choice = result.json.return_value["choices"][0]
    choice["finish_reason"] = finish
    choice["message"]["tool_calls"] = [
        {"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}
    ]
    return result


class LLMClientTests(unittest.TestCase):
    def client(self, *, require_constraints=False, **config):
        self.session = Mock()
        self.session.post.return_value = response()
        return LLMClient(
            {"provider": "gpt", "model": "test-model", **config},
            "test-secret-key", "Navigation skill.", require_constraints=require_constraints,
            session=self.session,
        )

    def deepseek(self, **options):
        return self.client(provider="deepseek", base_url="https://api.deepseek.com/beta",
                           require_constraints=True, request_options=options)

    def payload(self):
        return self.session.post.call_args.kwargs["json"]

    def test_gpt_receives_only_the_supplied_observation_and_task(self):
        client = self.client(request_options={"max_completion_tokens": 500})
        result = client.act(TEXT)
        self.assertEqual(result.action, "left")
        args, kwargs = self.session.post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-secret-key")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertFalse(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["max_completion_tokens"], 500)
        messages = kwargs["json"]["messages"]
        self.assertIn("Navigation skill.", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": TEXT.prompt})
        self.assertNotIn("tools", kwargs["json"])
        self.assertNotIn("response_format", kwargs["json"])
        self.assertNotIn("test-secret-key", str(result.metadata))
        self.assertNotIn("PRIVATE REASONING", str(result.metadata))
        self.assertEqual(result.metadata["usage"]["total_tokens"], 85)
        self.assertEqual(result.metadata["api_calls"], 1)

    def test_deepseek_text_and_budget(self):
        client = self.client(provider="deepseek", request_options={"thinking": {"type": "disabled"},
                                                                   "max_tokens": 64})
        self.session.post.return_value = response("  open door\n")
        self.assertEqual(client.act(TEXT).action, "open door")
        args, kwargs = self.session.post.call_args
        self.assertEqual(args[0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(kwargs["json"]["thinking"], {"type": "disabled"})
        self.assertNotIn("tools", kwargs["json"])

    def test_rejection_feedback_belongs_to_the_given_call(self):
        client = self.client()
        feedback = (Feedback("wait", "unavailable", "Please select an open road."),)
        self.assertEqual(client.act(TEXT, feedback).action, "left")
        self.assertEqual(self.payload()["messages"][-2:], [
            {"role": "assistant", "content": "wait"},
            {"role": "user", "content": "Please select an open road."},
        ])
        self.session.post.assert_called_once()
        self.assertEqual(client.act(TEXT).action, "left")
        self.assertEqual(len(self.payload()["messages"]), 2)

    def test_domain_specific_wire_names_are_supplied_by_the_observation(self):
        spec = ActionSpec(field="move", tool="choose_move", text_mode="uci")
        observation = Observation("chess-1", "White to move.", spec, ("e2e4",))
        for provider in ("gpt", "deepseek"):
            with self.subTest(provider=provider):
                if provider == "gpt":
                    client = self.client()
                    self.session.post.return_value = response('{"move":"e2e4"}')
                else:
                    client = self.deepseek()
                    self.session.post.return_value = tool_response('{"move":"e2e4"}', "choose_move")
                self.assertEqual(client.act(observation).action, "e2e4")
                if provider == "deepseek":
                    self.assertEqual(self.payload()["tools"][0]["function"]["name"], "choose_move")
                else:
                    schema = self.payload()["response_format"]["json_schema"]["schema"]
                    self.assertEqual(schema["required"], ["move"])
                self.session.post.return_value = response("e2e5")
                no_choices = Observation("chess-2", "Black to move.", spec)
                self.assertEqual(client.act(no_choices).metadata["output_mode"], "uci")
                self.assertNotIn("response_format", self.payload())
                self.assertNotIn("tools", self.payload())

    def test_gpt_schema_tracks_each_set_of_choices(self):
        client = self.client()
        self.session.post.return_value = response('{"action":"left"}')
        self.assertEqual(client.act(CHOICES).action, "left")
        schema = self.payload()["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["action"]["enum"], ["left", "right"])
        self.assertFalse(schema["schema"]["additionalProperties"])

        obs = Observation("p2", "You reach a bridge.", choices=("cross", "back"))
        with self.assertRaises(AgentOutputError) as context:
            client.act(obs)
        schema = self.payload()["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["action"]["enum"], ["cross", "back"])
        self.assertEqual(context.exception.metadata["error"], "constraint_violation")

    def test_deepseek_forces_tool_when_thinking_disabled(self):
        client = self.deepseek(thinking={"type": "disabled"}, reasoning_effort="high", max_tokens=1000)
        self.session.post.return_value = tool_response()
        decision = client.act(CHOICES)
        self.assertEqual(decision.action, "left")
        self.assertEqual(self.session.post.call_args.args[0], "https://api.deepseek.com/beta/chat/completions")
        payload = self.payload()
        function = payload["tools"][0]["function"]
        self.assertTrue(function["strict"])
        self.assertEqual(function["parameters"]["properties"]["action"]["enum"], ["left", "right"])
        self.assertEqual(payload["tool_choice"], {"type": "function", "function": {"name": "choose_action"}})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(decision.metadata["finish_reason"], "tool_calls")
        self.assertNotIn("PRIVATE REASONING", str(decision.metadata))

    def test_deepseek_thinking_preserves_budget_and_requires_a_tool_result(self):
        for options in ({}, {"thinking": {"type": "enabled"}, "reasoning_effort": "high", "max_tokens": 100000}):
            with self.subTest(options=options):
                client = self.deepseek(**options)
                self.session.post.return_value = response("left")
                with self.assertRaises(AgentOutputError) as context:
                    client.act(CHOICES)
                self.assertEqual(context.exception.metadata["error"], "missing_tool_call")
                self.session.post.assert_called_once()
                self.session.post.return_value = tool_response()
                self.assertEqual(client.act(CHOICES).action, "left")
                self.assertEqual(self.payload()["tool_choice"], "auto")
                for key, value in options.items():
                    self.assertEqual(self.payload()[key], value)

    def test_strict_tools_validate_arguments_name_and_count(self):
        for arguments, name in (("not JSON", "choose_action"), ('{"action":"stop"}', "choose_action"),
                                ('{"action":"left","extra":1}', "choose_action"),
                                ('{"action":"left"}', "other_tool"), ('{"action":4}', "choose_action")):
            with self.subTest(arguments=arguments, name=name):
                client = self.deepseek()
                self.session.post.return_value = tool_response(arguments, name)
                with self.assertRaises(AgentOutputError) as context:
                    client.act(CHOICES)
                self.assertEqual(context.exception.metadata["error"], "constraint_violation")
        client = self.deepseek()
        answer = tool_response()
        calls = answer.json.return_value["choices"][0]["message"]["tool_calls"]
        calls.append(dict(calls[0]))
        self.session.post.return_value = answer
        with self.assertRaises(AgentOutputError):
            client.act(CHOICES)

    def test_output_errors_keep_token_usage_and_finish_reason(self):
        cases = [(None, None, "empty_response"), ("", None, "empty_response"),
                 ("   ", "stop", "empty_response"), ("left", "length", "output_truncated"),
                 (None, "content_filter", "output_refused"), (None, "aborted", "output_interrupted")]
        for content, finish, expected in cases:
            with self.subTest(expected=expected):
                client = self.client()
                answer = response(content)
                answer.json.return_value["choices"][0]["finish_reason"] = finish
                self.session.post.return_value = answer
                with self.assertRaises(AgentOutputError) as context:
                    client.act(TEXT)
                metadata = context.exception.metadata
                self.assertEqual(metadata["error"], expected)
                self.assertEqual(metadata["finish_reason"], finish)
                self.assertEqual(metadata["usage"]["total_tokens"], 85)
                self.assertEqual(metadata["api_calls"], 1)
                self.session.post.assert_called_once()

    def test_truncated_tool_is_an_output_error_even_if_the_json_looks_valid(self):
        client = self.deepseek()
        self.session.post.return_value = tool_response(finish="length")
        with self.assertRaises(AgentOutputError) as context:
            client.act(CHOICES)
        self.assertEqual(context.exception.metadata["error"], "output_truncated")

    def test_failed_api_calls_do_not_leak_error_text_or_retry(self):
        client = self.deepseek()
        self.session.post.return_value = response(status=401)
        self.session.post.return_value.text = "test-secret-key private payload"
        with self.assertRaises(AgentError) as context:
            client.act(CHOICES)
        self.assertIn("HTTP 401", str(context.exception))
        self.assertNotIn("test-secret-key", str(context.exception))
        self.assertEqual(context.exception.metadata["error"], "api_error")
        self.assertEqual(context.exception.metadata["api_calls"], 1)
        self.session.post.assert_called_once()

        self.session.post.side_effect = requests.Timeout("test-secret-key")
        with self.assertRaises(AgentError) as context:
            client.act(TEXT)
        self.assertNotIn("test-secret-key", str(context.exception))

    def test_missing_choices_is_an_api_error(self):
        client = self.client()
        self.session.post.return_value.json.return_value = {"choices": []}
        with self.assertRaises(AgentError) as context:
            client.act(TEXT)
        self.assertEqual(context.exception.metadata["error"], "api_response_format")

    def test_redacts_final_outputs_and_discards_untrusted_usage_fields(self):
        client = self.client()
        answer = response("test-secret-key")
        answer.json.return_value["usage"]["private"] = "secret provider data"
        self.session.post.return_value = answer
        decision = client.act(TEXT)
        self.assertEqual(decision.metadata["raw_response"], "[REDACTED]")
        self.assertNotIn("secret provider", str(decision.metadata))
        client = self.deepseek()
        self.session.post.return_value = tool_response('{"action":"test-secret-key"}')
        with self.assertRaises(AgentOutputError) as context:
            client.act(CHOICES)
        self.assertNotIn("test-secret-key", str(context.exception.metadata))
        self.assertIn("[REDACTED]", str(context.exception.metadata))

    def test_configuration_cannot_override_request_or_authentication(self):
        for options in ({"messages": []}, {"model": "other"}, {"stream": True},
                        {"api_key": "secret"}, {"tools": []}, {"thinking": {"type": "enable"}},
                        {"max_tokens": 20, "max_completion_tokens": 20}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.client(request_options=options)

    def test_deepseek_beta_is_checked_at_configuration_and_for_actual_observations(self):
        with self.assertRaises(ValueError):
            self.client(provider="deepseek", require_constraints=True)
        client = self.client(provider="deepseek")
        with self.assertRaises(AgentError) as context:
            client.act(CHOICES)
        self.assertEqual(context.exception.metadata["api_calls"], 0)
        self.session.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
