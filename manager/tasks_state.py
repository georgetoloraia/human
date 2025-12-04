from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import json

from .tasks import Task, tasks_by_plugin


TASK_STATE_FILE = Path("manager/tasks_state.json")
TASK_STATE_VERSION = "1.0"


class TaskStateManager:
    """
    Tracks per-task performance:
      - passes / fails
      - streak of successes
      - last_status ("unknown"/"passing"/"failing")
      - last_error_type (e.g. "OK", "AssertionError", ...)
    """

    def __init__(self, tasks: Dict[str, Task]) -> None:
        self.tasks = tasks
        self.by_plugin = tasks_by_plugin(tasks)
        self.state: Dict[str, Any] = {}

        if TASK_STATE_FILE.exists():
            try:
                loaded = json.loads(TASK_STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.state = loaded
            except Exception:
                self.state = {}

        self.state.setdefault("_version", TASK_STATE_VERSION)

        for name, task in tasks.items():
            s = self.state.setdefault(
                name,
                {
                    "passes": 0,
                    "fails": 0,
                    "streak": 0,
                    "last_status": "unknown",
                    "last_error_type": "unknown",
                    "phase": getattr(task, "phase", 1),
                    "difficulty": getattr(task, "difficulty", 1),
                    "category": getattr(task, "category", "general"),
                    "plugin": getattr(task, "target_plugin", None),
                },
            )
            # refresh metadata when tasks change over time
            s.setdefault("phase", getattr(task, "phase", 1))
            s.setdefault("difficulty", getattr(task, "difficulty", 1))
            s.setdefault("category", getattr(task, "category", "general"))
            s.setdefault("plugin", getattr(task, "target_plugin", None))

        # drop tasks no longer present
        stale = [k for k in self.state.keys() if k not in tasks and not k.startswith("_")]
        for k in stale:
            self.state.pop(k, None)

        self._save()

    def _save(self) -> None:
        TASK_STATE_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def record_plugin_result(self, plugin_name: str, success: bool, error_type: str) -> None:
        for tname in self.by_plugin.get(plugin_name, []):
            self._record_task_result(tname, success, error_type)
        self._save()

    def _record_task_result(self, task_name: str, success: bool, error_type: str) -> None:
        task_obj = self.tasks.get(task_name)
        s = self.state.setdefault(
            task_name,
            {
                "passes": 0,
                "fails": 0,
                "streak": 0,
                "last_status": "unknown",
                "last_error_type": "unknown",
                "phase": getattr(task_obj, "phase", 1) if task_obj else 1,
                "difficulty": getattr(task_obj, "difficulty", 1) if task_obj else 1,
                "category": getattr(task_obj, "category", "general") if task_obj else "general",
                "plugin": getattr(task_obj, "target_plugin", None) if task_obj else None,
            },
        )
        if success:
            s["passes"] += 1
            s["streak"] = s["streak"] + 1 if s.get("streak", 0) > 0 else 1
            s["last_status"] = "passing"
            s["last_error_type"] = error_type or "OK"
        else:
            s["fails"] += 1
            s["streak"] = s["streak"] - 1 if s.get("streak", 0) < 0 else -1
            s["last_status"] = "failing"
            s["last_error_type"] = error_type or "Other"

    def record_task_results(self, results: Dict[str, tuple[bool, str]]) -> None:
        """
        Record per-task outcomes in bulk.
        """
        for tname, (success, err) in results.items():
            self._record_task_result(tname, success, err)
        self._save()

    def plugin_task_stats(self, plugin_name: str) -> Dict[str, float]:
        task_names = self.by_plugin.get(plugin_name, [])
        if not task_names:
            return {
                "task_count": 0.0,
                "avg_streak": 0.0,
                "failing_count": 0.0,
            }
        streaks = []
        failing = 0
        for tname in task_names:
            s = self.state.get(tname, {})
            streaks.append(float(s.get("streak", 0)))
            if s.get("last_status") == "failing":
                failing += 1
        avg_streak = sum(streaks) / len(streaks) if streaks else 0.0
        return {
            "task_count": float(len(task_names)),
            "avg_streak": avg_streak,
            "failing_count": float(failing),
        }

    def summary(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for tname, task in self.tasks.items():
            s = self.state.get(tname, {})
            items.append(
                {
                    "name": tname,
                    "plugin": task.target_plugin,
                    "streak": s.get("streak", 0),
                    "passes": s.get("passes", 0),
                    "fails": s.get("fails", 0),
                    "last_status": s.get("last_status", "unknown"),
                    "last_error_type": s.get("last_error_type", "unknown"),
                }
            )
        return items

    def get_weakest_task(self):
        weakest = None
        worst = float("inf")
        for name, info in self.state.items():
            if name.startswith("_"):
                continue
            score = float(info.get("streak", 0)) - float(info.get("passes", 0))
            if score < worst:
                worst = score
                weakest = name
        return weakest
