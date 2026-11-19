"""GPT and DeepSeek agents for text observations and optional action choices."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import time
from typing import Any
from urllib.parse import urlsplit

import requests

from .contracts import AgentError, AgentOutputError, Decision, Feedback, Observation


OUTPUT_ERRORS = {
    "output_truncated": "LLM 输出被截断（finish_reason=length）；请检查推理及输出 token 预算。",
    "output_refused": "LLM 拒绝回答或输出被过滤。",
    "output_interrupted": "LLM 服务端中断生成，未获得完整动作。",
    "empty_response": "LLM 最终回复为空，请检查推理预算和 finish_reason。",
    "missing_tool_call": "LLM 没有返回要求的工具调用。",
    "constraint_violation": "LLM 返回值不符合本次请求的输出约束。",
}


_OPTIONS = {
    "temperature", "top_p", "max_completion_tokens", "max_tokens",
    "reasoning_effort", "thinking", "seed", "stop",
}
_USAGE_COUNTS = {
    "prompt_tokens", "completion_tokens", "total_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    "cached_tokens", "audio_tokens", "reasoning_tokens",
    "accepted_prediction_tokens", "rejected_prediction_tokens",
}
_USAGE_DETAILS = {"prompt_tokens_details", "completion_tokens_details"}


def _usage_counts(value: Any) -> dict[str, Any]:
    """Keep token counters only, never arbitrary response metadata."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, count in value.items():
        if key in _USAGE_COUNTS and type(count) is int and count >= 0:
            result[key] = count
        elif key in _USAGE_DETAILS and isinstance(count, dict):
            result[key] = _usage_counts(count)
    return result


class LLMClient:
    def __init__(
        self,
        config: dict[str, Any],
        api_key: str,
        skill_text: str,
        *,
        require_constraints: bool = False,
        session: requests.Session | None = None,
    ):
        provider = config.get("provider", "gpt")
        if provider not in {"gpt", "deepseek"}:
            raise ValueError("LLM.provider 必须为 gpt 或 deepseek。")
        self.provider = provider
        defaults = {
            "gpt": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com",
        }
        self.base_url = str(config.get("base_url", defaults[provider])).rstrip("/")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("LLM.base_url 必须是不含凭证、查询参数的 HTTP(S) 地址。")
        self.model = config.get("model")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("请在 LLM.yaml 中填写 model。")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("LLM API key 不能为空。")
        if not isinstance(skill_text, str) or not skill_text.strip():
            raise ValueError("任务 skill 不能为空。")
        self._api_key = api_key.strip()
        self.skill_text = skill_text
        self.request_timeout = config.get("request_timeout", 120)
        if (isinstance(self.request_timeout, bool)
                or not isinstance(self.request_timeout, (int, float))
                or not math.isfinite(self.request_timeout) or self.request_timeout <= 0):
            raise ValueError("LLM.request_timeout 必须为正数。")
        if require_constraints:
            self._check_constraints_support()
        options = config.get("request_options", {})
        if not isinstance(options, dict):
            raise ValueError("LLM.request_options 必须是 YAML 字典。")
        if any(key not in _OPTIONS for key in options):
            raise ValueError("LLM.request_options 包含不支持的字段，请参考 README。")
        if "max_tokens" in options and "max_completion_tokens" in options:
            raise ValueError("max_tokens 和 max_completion_tokens 只能设置一个。")
        thinking = options.get("thinking")
        if thinking is not None and (
            not isinstance(thinking, dict) or set(thinking) != {"type"}
            or thinking["type"] not in ("enabled", "disabled")
        ):
            raise ValueError("thinking 必须为 {type: enabled} 或 {type: disabled}，注意 enabled 的拼写。")
        self.thinking_enabled = (
            thinking["type"] == "enabled" if thinking is not None
            else options.get("reasoning_effort") != "none"
        )
        self.request_options = deepcopy(options)
        self.session = session if session is not None else requests.Session()

    def close(self) -> None:
        self.session.close()

    def _check_constraints_support(self) -> None:
        if self.provider == "deepseek" and not urlsplit(self.base_url).path.endswith("/beta"):
            raise ValueError("DeepSeek 严格工具调用需要 Beta 地址，请将 base_url 设为 https://api.deepseek.com/beta。")

    def _request_format(self, observation: Observation) -> tuple[str, str, dict[str, Any]]:
        """Translate the environment's output contract into the provider's wire format."""
        if observation.choices is None:
            return observation.spec.text_mode, "Return only the selected action.", {}
        self._check_constraints_support()
        field, tool = observation.spec.field, observation.spec.tool
        schema = {
            "type": "object",
            "properties": {field: {"type": "string", "enum": list(observation.choices)}},
            "required": [field],
            "additionalProperties": False,
        }
        if self.provider == "gpt":
            instruction = f'Return only a JSON object with one key "{field}" set to an allowed value.'
            constraints = {"response_format": {"type": "json_schema", "json_schema": {
                "name": tool, "strict": True, "schema": schema,
            }}}
            return "json_schema", instruction, constraints
        instruction = f'Call {tool} exactly once, setting "{field}" to an allowed value.'
        constraints = {
            "tools": [{"type": "function", "function": {
                "name": tool, "description": "Select one action from the supplied choices.",
                "strict": True, "parameters": schema,
            }}],
            # DeepSeek permits forced tool choice only when thinking is disabled.
            "tool_choice": "auto" if self.thinking_enabled else {
                "type": "function", "function": {"name": tool},
            },
        }
        return "strict_tool", instruction, constraints

    def _complete(self, messages: list[dict[str, str]], constraints: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **self.request_options,
            **constraints,
        }
        try:
            response = self.session.post(
                self.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.request_timeout,
                allow_redirects=False,
            )
        except requests.RequestException:
            # Exception strings may contain URLs, headers, or response bodies.
            raise AgentError("LLM API 网络请求失败或超时；请检查网络及 request_timeout。") from None
        try:
            if not 200 <= response.status_code < 300:
                raise AgentError(f"LLM API 返回 HTTP {response.status_code}；请检查模型、参数、额度和 API key。")
            try:
                data = response.json()
            except ValueError:
                raise AgentError("LLM API 未返回有效 JSON。") from None
            if not isinstance(data, dict):
                raise AgentError("LLM API 返回的数据结构无效。")
            return data
        finally:
            response.close()

    def act(self, observation: Observation, feedback: tuple[Feedback, ...] = ()) -> Decision:
        try:
            mode, instruction, constraints = self._request_format(observation)
        except ValueError as exc:
            raise AgentError(str(exc), {"error": "api_configuration", "api_calls": 0}) from None
        messages = [
            {"role": "system", "content": self.skill_text + "\n\n" + instruction},
            {"role": "user", "content": observation.prompt},
        ]
        for rejection in feedback:
            messages.extend([
                {"role": "assistant", "content": rejection.action},
                {"role": "user", "content": rejection.message},
            ])
        metadata = {
            "api_calls": 1, "output_mode": mode, "raw_response": None,
            "finish_reason": None, "usage": {},
        }
        started = time.perf_counter()
        try:
            data = self._complete(messages, constraints)
        except AgentError as exc:
            metadata.update(error="api_error", elapsed_seconds=round(time.perf_counter() - started, 6))
            raise AgentError(str(exc), metadata) from None
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        metadata["usage"] = _usage_counts(data.get("usage"))
        try:
            choice = data["choices"][0]
            message = choice["message"]
            raw = message.get("content")
            finish = choice.get("finish_reason")
            if raw is not None and not isinstance(raw, str):
                raise TypeError
            if finish is not None and not isinstance(finish, str):
                raise TypeError
        except (KeyError, IndexError, TypeError, AttributeError):
            metadata["error"] = "api_response_format"
            raise AgentError("LLM API 缺少有效的 choices[0].message。", metadata) from None

        # Only final model outputs are logged, never reasoning or request headers.
        if raw is not None:
            metadata["raw_response"] = raw.replace(self._api_key, "[REDACTED]")
        if finish:
            metadata["finish_reason"] = finish.replace(self._api_key, "[REDACTED]")
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            metadata["raw_tool_arguments"] = [
                call["function"]["arguments"].replace(self._api_key, "[REDACTED]")
                for call in tool_calls if isinstance(call, dict)
                and isinstance(call.get("function"), dict)
                and isinstance(call["function"].get("arguments"), str)
            ]

        def failure(code: str) -> AgentOutputError:
            return AgentOutputError(OUTPUT_ERRORS[code], {**metadata, "error": code})

        if finish == "length":
            raise failure("output_truncated")
        if message.get("refusal") or finish == "content_filter":
            raise failure("output_refused")
        if finish in {"aborted", "insufficient_system_resource"}:
            raise failure("output_interrupted")

        text = (raw or "").strip()
        if observation.choices is not None and self.provider == "deepseek":
            if not tool_calls:
                raise failure("missing_tool_call")
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                raise failure("constraint_violation")
            try:
                call = tool_calls[0]
                function = call["function"]
                if call["type"] != "function" or function["name"] != observation.spec.tool:
                    raise ValueError
                text = function["arguments"]
                if not isinstance(text, str):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise failure("constraint_violation") from None
        elif not text:
            raise failure("empty_response")

        if observation.choices is not None:
            field = observation.spec.field
            try:
                answer = json.loads(text)
                if (not isinstance(answer, dict) or set(answer) != {field}
                        or not isinstance(answer[field], str) or answer[field] not in observation.choices):
                    raise ValueError
                text = answer[field]
            except (ValueError, TypeError):
                raise failure("constraint_violation") from None
        return Decision(text, metadata)
