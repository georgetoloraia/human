from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

from manager.agents import BaseAgent
from manager.value_function import ValueFunction


class PlannerAgent(BaseAgent):
    """
    Chooses the next plugins/tasks to focus on, blending drive scores with value estimates.
    """

    def __init__(self, value_function: ValueFunction, max_targets: Optional[int] = None) -> None:
        super().__init__()
        self.value_function = value_function
        self.max_targets = max_targets or int(os.getenv("HUMAN_PLANNER_MAX_TARGETS", "2"))
        self.epsilon = float(os.getenv("HUMAN_PLANNER_EPSILON", "0.1"))

    def act(self) -> Dict[str, object]:
        plugin_scores: Dict[str, float] = self.state.get("plugin_scores", {})  # type: ignore[assignment]
        candidates: List[str] = list(plugin_scores.keys())
        if not candidates:
            return {"target_plugins": [], "strategy": "mutation"}

        combined: List[tuple[float, str]] = []
        for name in candidates:
            base = float(plugin_scores.get(name, 0.0))
            vf_score = self.value_function.score(f"plugin:{name}", candidate_type="plugin", strategy="planner")
            combined.append((base + vf_score, name))
        combined.sort(key=lambda x: x[0], reverse=True)

        picks: List[str] = []
        for score, name in combined:
            if random.random() < self.epsilon:
                picks.append(random.choice(candidates))
            else:
                picks.append(name)
            if len(picks) >= max(1, self.max_targets):
                break

        return {
            "target_plugins": picks,
            "strategy": "mutation",
            "scores": {n: s for s, n in combined},
            "exploration": self.epsilon,
        }
