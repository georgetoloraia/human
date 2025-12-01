from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from manager.memory import BrainMemory

META_STATE_FILE = Path("manager/meta_policy_state.json")


class MetaPolicy:
    """
    Simple meta-learner that experiments with learning_policy parameters and
    keeps changes that improve learning speed.
    """

    def __init__(self, brain: BrainMemory, window: int = 20, trial_len: int = 10):
        self.enabled = os.getenv("HUMAN_META_LEARN") == "1"
        self.brain = brain
        self.window = window
        self.trial_len = trial_len
        self.state: Dict[str, Any] = {
            "status": "baseline",  # baseline | candidate
            "current_policy": self.brain.get_learning_policy(),
            "prev_policy": None,
            "experiment": None,
            "baseline_speed": 0.0,
            "last_age": None,
            "last_skill": None,
            "candidate_start_age": None,
            "candidate_start_skill": None,
            "experiments": [],
            "step_counter": 0,
        }
        if META_STATE_FILE.exists():
            try:
                loaded = json.loads(META_STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.state.update(loaded)
            except Exception:
                pass
        self._save()

    def _save(self) -> None:
        META_STATE_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _mutate_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        new_policy = dict(policy)
        weights = dict(new_policy.get("weights", {}))
        new_policy["weights"] = weights
        choices = ["exploration_depth", "max_plugins_per_step", "web_consult_threshold"]
        param = choices[self.state["step_counter"] % len(choices)]
        if param == "exploration_depth":
            val = int(new_policy.get("exploration_depth", 3))
            new_policy["exploration_depth"] = max(1, min(10, val + 1))
        elif param == "max_plugins_per_step":
            val = int(new_policy.get("max_plugins_per_step", 2))
            new_policy["max_plugins_per_step"] = max(1, min(4, val + 1))
        elif param == "web_consult_threshold":
            val = int(new_policy.get("web_consult_threshold", 3))
            new_policy["web_consult_threshold"] = max(1, min(10, val + 1))
        return new_policy

    def tick(self, age: int, skill: float) -> Dict[str, Any]:
        status = {"enabled": self.enabled, "mode": self.state.get("status"), "experiment": None}
        if not self.enabled:
            return status
        self.state["step_counter"] = self.state.get("step_counter", 0) + 1
        last_age = self.state.get("last_age")
        last_skill = self.state.get("last_skill")
        if last_age is None or last_skill is None:
            self.state["last_age"] = age
            self.state["last_skill"] = skill
            self._save()
            return status
        delta_age = max(1, age - last_age)
        delta_skill = skill - last_skill
        speed = delta_skill / delta_age
        self.state["last_age"] = age
        self.state["last_skill"] = skill

        if self.state["status"] == "baseline":
            # update baseline speed (EMA)
            base = float(self.state.get("baseline_speed", 0.0))
            self.state["baseline_speed"] = 0.7 * base + 0.3 * speed
            # maybe start experiment
            if self.state["step_counter"] % self.window == 0:
                prev_policy = self.brain.get_learning_policy()
                candidate = self._mutate_policy(prev_policy)
                self.state["prev_policy"] = prev_policy
                self.state["current_policy"] = candidate
                self.brain.set_learning_policy(candidate)
                self.state["status"] = "candidate"
                self.state["candidate_start_age"] = age
                self.state["candidate_start_skill"] = skill
                self.state["experiment"] = {
                    "id": len(self.state.get("experiments", [])) + 1,
                    "param": "auto",
                    "old": prev_policy,
                    "new": candidate,
                    "start_age": age,
                }
                status["mode"] = "candidate"
                status["experiment"] = self.state["experiment"]
        elif self.state["status"] == "candidate":
            start_age = self.state.get("candidate_start_age", age)
            start_skill = self.state.get("candidate_start_skill", skill)
            if age - start_age >= self.trial_len:
                candidate_speed = (skill - start_skill) / max(1, age - start_age)
                baseline_speed = float(self.state.get("baseline_speed", 0.0))
                accepted = candidate_speed > baseline_speed
                exp = self.state.get("experiment", {})
                exp.update(
                    {
                        "end_age": age,
                        "baseline_speed": baseline_speed,
                        "candidate_speed": candidate_speed,
                        "outcome": "accepted" if accepted else "reverted",
                    }
                )
                self.state.setdefault("experiments", []).append(exp)
                if accepted:
                    self.state["baseline_speed"] = candidate_speed
                else:
                    prev = self.state.get("prev_policy") or self.brain.get_learning_policy()
                    self.brain.set_learning_policy(prev)
                    self.state["current_policy"] = prev
                # reset trial
                self.state["status"] = "baseline"
                self.state["experiment"] = None
                self.state["prev_policy"] = None
                self.state["candidate_start_age"] = None
                self.state["candidate_start_skill"] = None
                status["mode"] = "baseline"
                status["experiment"] = exp
        self._save()
        return status
