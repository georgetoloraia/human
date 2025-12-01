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
