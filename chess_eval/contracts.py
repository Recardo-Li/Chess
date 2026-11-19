"""Interfaces shared by the evaluation loop, task environments and agents."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ActionSpec:
    """Wire names used by a task's skill and its structured output protocol."""

    field: str = "action"
    tool: str = "choose_action"
    text_mode: str = "text"


@dataclass(frozen=True)
class Observation:
    """Public input at one decision point. choices, if present, is a strict enum."""

    state_id: str
    prompt: str
    spec: ActionSpec = ActionSpec()
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Feedback:
    """A rejected action and the information the agent may use on its next try."""

    action: str
    code: str
    message: str


@dataclass(frozen=True)
class Decision:
    action: str
    metadata: dict = field(default_factory=dict)


class AgentError(RuntimeError):
    def __init__(self, message: str, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


class AgentOutputError(AgentError):
    """The agent received an unusable or incomplete response."""


class EnvironmentError(RuntimeError):
    """The environment could not continue or confirm an operation."""


@dataclass(frozen=True)
class Turn:
    """Only observation is shared with the agent. Other fields are for the log."""

    observation: Observation
    log_fields: dict = field(default_factory=dict)
    action_key: str = "action"
    invalid_outcome: str = "invalid_action"


@dataclass(frozen=True)
class EpisodeResult:
    outcome: str
    result: str = "*"
    score: float | None = None
    reason: str | None = None


class Agent(Protocol):
    def act(self, observation: Observation, feedback: tuple[Feedback, ...] = ()) -> Decision:
        """Propose one action. Include any reportable call statistics in metadata."""
        ...


class Environment(Protocol):
    def turns(self, log: Callable[[dict], None]) -> Generator[Turn | EpisodeResult, None, None]:
        """Open the episode. Yield only decision points, then one terminal result."""
        ...

    def validate(self, state_id: str, action: str) -> Feedback | None:
        """Check an action without changing state. A stale state is an environment error."""
        ...

    def submit(self, state_id: str, action: str) -> None:
        """Apply a validated action to its original state exactly once."""
        ...

    def finish(self, output_dir: Path, outcome: EpisodeResult, agent_name: str) -> dict:
        """Release the episode, write task-specific artifacts and return summary fields."""
        ...
