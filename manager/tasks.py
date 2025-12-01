from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


TASKS_DIR = Path("tasks")


@dataclass
class Task:
    name: str
    target_plugin: str
    target_function: Optional[str]
    description: str
    requirements: List[str]


def load_tasks() -> Dict[str, Task]:
    tasks: Dict[str, Task] = {}
    if not TASKS_DIR.exists():
        return tasks
    for path in TASKS_DIR.glob("*.yml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name")
        target_plugin = data.get("target_plugin")
        target_function = data.get("target_function")
        description = data.get("description", "")
        requirements = data.get("requirements") or []
        if not name or not target_plugin:
            continue
        tasks[name] = Task(
            name=name,
            target_plugin=target_plugin,
            target_function=target_function,
            description=description,
            requirements=list(requirements),
        )
    return tasks


def tasks_by_plugin(tasks: Dict[str, Task]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for tname, task in tasks.items():
        mapping.setdefault(task.target_plugin, []).append(tname)
    return mapping
