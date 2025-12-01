#!/usr/bin/env python3
from pathlib import Path
import shutil
import json

ROOT = Path(__file__).resolve().parents[1]
STATE_FILES = [
    ROOT / "manager" / "brain_memory.json",
    ROOT / "manager" / "tasks_state.json",
    ROOT / "manager" / "concept_graph.json",
    ROOT / "manager" / "curriculum_state.json",
    ROOT / "manager" / "mind_diary.json",
    ROOT / "manager" / "metrics.json",
]

SAMPLE_PLUGIN = ROOT / "plugins" / "sample_plugin.py"
SAMPLE_TASK = ROOT / "tasks" / "list_sum.yml"


def reset_state():
    for f in STATE_FILES:
        if f.exists():
            f.unlink()
    # recreate minimal plugin/task if missing
    SAMPLE_PLUGIN.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_TASK.parent.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_PLUGIN.exists():
        SAMPLE_PLUGIN.write_text(
            "def process(values):\n"
            "    total = 0\n"
            "    for v in values:\n"
            "        total += v\n"
            "    return total\n",
            encoding="utf-8",
        )
    if not SAMPLE_TASK.exists():
        SAMPLE_TASK.write_text(
            "name: list_sum\n"
            "phase: 1\n"
            "target_plugin: sample_plugin.py\n"
            "target_function: process\n"
            "description: The function must return the sum of a list of integers.\n"
            "requirements:\n"
            "  - process([1,2,3]) == 6\n"
            "  - process([]) == 0\n"
            "  - process([-1,1]) == 0\n",
            encoding="utf-8",
        )
    print("State reset. Next run will start fresh.")


if __name__ == "__main__":
    reset_state()
