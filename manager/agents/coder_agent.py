from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from manager.agents import BaseAgent

if TYPE_CHECKING:  # pragma: no cover
    from manager.mind import Mind


class CoderAgent(BaseAgent):
    """
    Executes code mutations and tests based on planner output.
    """

    def __init__(self, mind: "Mind") -> None:
        super().__init__()
        self.mind = mind

    def act(self) -> Dict[str, object]:
        plan = self.state.get("plan", {})  # type: ignore[assignment]
        targets: Optional[List[str]] = plan.get("target_plugins") if isinstance(plan, dict) else None  # type: ignore[assignment]
        active_tasks: List[str] = self.state.get("active_tasks", [])  # type: ignore[assignment]
        strategy = plan.get("strategy") if isinstance(plan, dict) else None

        # reuse the Mind's existing mutation pipeline while constraining targets
        preserved_actions = list(self.mind._current_step_actions)
        preserved_selection = list(self.mind._last_selection_info)
        self.mind._current_step_actions = []
        self.mind._last_selection_info = []
        self.mind._act_and_learn(active_tasks, forced_targets=targets, strategy=strategy)
        merged_actions = preserved_actions + list(self.mind._current_step_actions)
        merged_selection = preserved_selection + list(self.mind._last_selection_info)
        self.mind._current_step_actions = merged_actions
        self.mind._last_selection_info = merged_selection

        # Debug trace to spot idle behavior.
        print("[CoderAgent] plan:", plan)
        print("[CoderAgent] targets:", targets)
        print("[CoderAgent] active_tasks:", active_tasks)
        print("[CoderAgent] new actions:", len(merged_actions) - len(preserved_actions))
        return {
            "actions": merged_actions,
            "selection": merged_selection,
            "strategy": strategy or "mutation",
        }
