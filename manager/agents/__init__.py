from __future__ import annotations

from typing import Dict, Optional


class BaseAgent:
    """
    Minimal interface shared by all agents.
    """

    def __init__(self) -> None:
        self.state: Dict[str, object] = {}

    def observe(self, state: Dict[str, object]) -> None:  # pragma: no cover - simple setter
        self.state = state or {}

    def act(self) -> Optional[Dict[str, object]]:  # pragma: no cover - to be overridden
        raise NotImplementedError("act must be implemented by subclasses")

    def receive_feedback(self, feedback: Dict[str, object]) -> None:  # pragma: no cover - optional override
        # Agents can override to adjust internal caches; default is no-op
        return


from .planner_agent import PlannerAgent  # noqa: E402  # pragma: no cover
from .coder_agent import CoderAgent  # noqa: E402  # pragma: no cover
from .critic_agent import CriticAgent  # noqa: E402  # pragma: no cover
from .memory_agent import MemoryAgent  # noqa: E402  # pragma: no cover
