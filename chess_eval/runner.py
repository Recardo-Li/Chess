"""Run and record an episode through the observation/action interfaces."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .contracts import (
    Agent, AgentError, AgentOutputError, Environment, EnvironmentError,
    EpisodeResult, Feedback, Turn,
)


class GameRunner:
    def __init__(
        self, environment: Environment, agent: Agent, *, output_dir: Path,
        agent_name: str, skill_text: str, settings: dict, invalid_action_retries: int = 0,
    ):
        if type(invalid_action_retries) is not int or invalid_action_retries < 0:
            raise ValueError("invalid_action_retries 必须为非负整数。")
        self.environment = environment
        self.agent = agent
        self.agent_name = agent_name
        self.skill_text = skill_text
        self.settings = settings
        self.invalid_action_retries = invalid_action_retries
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output_dir = output_dir / f"{stamp}-{uuid4().hex[:8]}"
        self.attempts: list[dict] = []

    def log(self, event: dict) -> None:
        with (self.output_dir / "moves.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _act(self, turn: Turn) -> EpisodeResult | None:
        observation = turn.observation
        feedback: list[Feedback] = []
        attempts: list[dict] = []
        event = {**turn.log_fields, "type": "decision", "state_id": observation.state_id,
                 "attempts": attempts}
        try:
            for number in range(1, self.invalid_action_retries + 2):
                try:
                    decision = self.agent.act(observation, tuple(feedback))
                except AgentError as exc:
                    status = "output_error" if isinstance(exc, AgentOutputError) else "agent_error"
                    attempts.append({**exc.metadata, "attempt": number, "valid": False,
                                     "error": exc.metadata.get("error") or status, "status": status})
                    raise
                attempt = {**decision.metadata, "attempt": number, "valid": False,
                           "error": None, "status": "unvalidated"}
                attempts.append(attempt)
                rejected = self.environment.validate(observation.state_id, decision.action)
                if rejected is None:
                    attempt.update(valid=True, status="valid")
                    event[turn.action_key] = decision.action
                    break
                attempt.update(error=rejected.code, status="invalid_action")
                feedback.append(rejected)
            else:
                return EpisodeResult(turn.invalid_outcome, score=0, reason=(
                    f"动作未通过环境校验（{feedback[-1].code}），已用完 {len(attempts)} 次尝试。"
                ))
        finally:
            self.attempts.extend(attempts)
            self.log(event)

        # Persist the decision before submitting, even when the response is lost.
        self.environment.submit(observation.state_id, decision.action)
        return None

    def _play(self) -> EpisodeResult:
        with closing(self.environment.turns(self.log)) as turns:
            for item in turns:
                if isinstance(item, EpisodeResult):
                    return item
                outcome = self._act(item)
                if outcome is not None:
                    return outcome
        raise EnvironmentError("环境在报告终局前关闭。")

    def _metrics(self) -> dict:
        valid = sum(a["status"] == "valid" for a in self.attempts)
        invalid = sum(a["status"] == "invalid_action" for a in self.attempts)
        return {
            "agent_attempts": len(self.attempts),
            "api_calls": sum(a.get("api_calls", 0) for a in self.attempts),
            "legal_responses": valid,
            "invalid_responses": invalid,
            "output_error_responses": sum(a["status"] == "output_error" for a in self.attempts),
            "legal_response_rate": valid / (valid + invalid) if valid + invalid else None,
            "llm_seconds": round(sum(a.get("elapsed_seconds", 0) for a in self.attempts if a.get("api_calls")), 6),
            "total_tokens": sum(a.get("usage", {}).get("total_tokens", 0) for a in self.attempts),
        }

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True)
        started_at = datetime.now(timezone.utc).isoformat()
        (self.output_dir / "skill.md").write_text(self.skill_text, encoding="utf-8")
        outcome = EpisodeResult("incomplete")
        try:
            outcome = self._play()
        except AgentOutputError as exc:
            outcome = EpisodeResult("output_error", reason=str(exc))
        except (AgentError, EnvironmentError) as exc:
            outcome = EpisodeResult("error", reason=str(exc))
        except KeyboardInterrupt:
            outcome = EpisodeResult("interrupted", reason="用户中断。")
        finally:
            fields = self.environment.finish(self.output_dir, outcome, self.agent_name)
            summary = {
                **fields, **asdict(outcome), **self._metrics(),
                "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
                "settings": self.settings,
                "skill_sha256": hashlib.sha256(self.skill_text.encode("utf-8")).hexdigest(),
            }
            if not outcome.reason:
                del summary["reason"]
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        if outcome.reason:
            print(outcome.reason, flush=True)
        return summary
