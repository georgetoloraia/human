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

        for name in tasks.keys():
            self.state.setdefault(
                name,
                {
                    "passes": 0,
                    "fails": 0,
                    "streak": 0,
                    "last_status": "unknown",
                    "last_error_type": "unknown",
                },
            )

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
        s = self.state.setdefault(
            task_name,
            {
                "passes": 0,
                "fails": 0,
                "streak": 0,
                "last_status": "unknown",
                "last_error_type": "unknown",
            },
        )
        if success:
            s["passes"] += 1
            s["streak"] += 1
            s["last_status"] = "passing"
            s["last_error_type"] = error_type
        else:
            s["fails"] += 1
            s["streak"] = 0
            s["last_status"] = "failing"
            s["last_error_type"] = error_type

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
