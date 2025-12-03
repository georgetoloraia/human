from __future__ import annotations

from typing import Dict, List

from manager.agents import BaseAgent
from manager.reward import compute_reward
from manager.value_function import ValueFunction
from manager.metrics import Metrics


class CriticAgent(BaseAgent):
    """
    Evaluates coder outcomes, assigns rewards, and updates learning signals.
    """

    def __init__(self, neuron_graph, value_function: ValueFunction, metrics: Metrics) -> None:
        super().__init__()
        self.neuron_graph = neuron_graph
        self.value_function = value_function
        self.metrics = metrics

    def act(self) -> Dict[str, object]:
        actions: List[Dict[str, object]] = self.state.get("actions", [])  # type: ignore[assignment]
        plan = self.state.get("plan", {}) if isinstance(self.state, dict) else {}  # type: ignore[assignment]
        rewards: Dict[str, float] = {}
        strategy = plan.get("strategy") if isinstance(plan, dict) else None
        for action in actions:
            plugin = action.get("plugin")
            if not plugin:
                continue
            reward = compute_reward(action)
            rewards[plugin] = reward
        return {"reward_by_plugin": rewards, "strategy": strategy}
