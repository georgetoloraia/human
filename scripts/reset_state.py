#!/usr/bin/env python3
"""
Reset runtime state for the digital brain.

Backs up known JSON state files, then removes them so the next run starts fresh.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUPS_DIR = REPO_ROOT / "backups"

# State files to clear (relative to repo root)
STATE_PATHS = [
    REPO_ROOT / "manager" / "brain_memory.json",
    REPO_ROOT / "manager" / "concept_graph.json",
    REPO_ROOT / "manager" / "curriculum_state.json",
    REPO_ROOT / "manager" / "tasks_state.json",
    REPO_ROOT / "manager" / "mind_diary.json",
    REPO_ROOT / "manager" / "neuron_graph.json",
    REPO_ROOT / "manager" / "metrics.json",
    REPO_ROOT / "manager" / "traces" / "trace.jsonl",
]


def ensure_backup_dir() -> Path:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = BACKUPS_DIR / stamp
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def reset_state() -> None:
    backup_dir = ensure_backup_dir()
    moved = []
    missing = []

    for path in STATE_PATHS:
        if path.exists():
            target = backup_dir / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            moved.append(path.relative_to(REPO_ROOT))
        else:
            missing.append(path.relative_to(REPO_ROOT))

    print(f"Backed up state to {backup_dir.relative_to(REPO_ROOT)}")
    if moved:
        print("Moved:")
        for p in moved:
            print(f"  - {p}")
    if missing:
        print("Not found (already clean):")
        for p in missing:
            print(f"  - {p}")


if __name__ == "__main__":
    reset_state()
