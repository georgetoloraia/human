from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


METRICS_FILE = Path("manager/metrics.json")
METRICS_VERSION = "1.0"


class Metrics:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "_version": METRICS_VERSION,
            "steps": 0,
            "mutations_accepted": 0,
            "mutations_rejected": 0,
            "web_consults": 0,
            "tasks_mastered": 0,
            "plugin_stats": {},
            "strategy_stats": {},
        }
        if METRICS_FILE.exists():
            try:
                loaded = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.state.update(loaded)
            except Exception:
                pass
        self._save()

    def _save(self) -> None:
        METRICS_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def step(self) -> None:
        self.state["steps"] = self.state.get("steps", 0) + 1
        self._save()

    def add_accepted(self, n: int = 1) -> None:
        self.state["mutations_accepted"] = self.state.get("mutations_accepted", 0) + n
        self._save()

    def add_rejected(self, n: int = 1) -> None:
        self.state["mutations_rejected"] = self.state.get("mutations_rejected", 0) + n
        self._save()

    def add_web_consult(self, n: int = 1) -> None:
        self.state["web_consults"] = self.state.get("web_consults", 0) + n
        self._save()

    def set_tasks_mastered(self, count: int) -> None:
        self.state["tasks_mastered"] = count
        self._save()

    def _update_stat_bucket(self, bucket: str, key: str, success: bool, reward: float) -> None:
        stats = self.state.setdefault(bucket, {}).setdefault(
            key,
            {"invocations": 0, "success": 0, "fail": 0, "reward_sum": 0.0},
        )
        stats["invocations"] = stats.get("invocations", 0) + 1
        stats["reward_sum"] = stats.get("reward_sum", 0.0) + float(reward)
        if success:
            stats["success"] = stats.get("success", 0) + 1
        else:
            stats["fail"] = stats.get("fail", 0) + 1
        self.state[bucket][key] = stats
        self._save()

    def record_plugin_outcome(self, plugin_name: str, success: bool, reward: float) -> None:
        self._update_stat_bucket("plugin_stats", plugin_name, success, reward)

    def record_strategy_outcome(self, strategy: str, success: bool, reward: float) -> None:
        self._update_stat_bucket("strategy_stats", strategy, success, reward)

    def average_reward(self, key: str, bucket: str = "plugin_stats") -> float:
        stats = self.state.get(bucket, {}).get(key, {})
        invocations = float(stats.get("invocations", 0))
        if invocations <= 0:
            return 0.0
        return float(stats.get("reward_sum", 0.0)) / invocations
