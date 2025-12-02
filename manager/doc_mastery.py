from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .concept_miner import task_name_for_concept, _extract_state  # type: ignore

DOC_MASTERY_FILE = Path("manager/doc_mastery_state.json")
MASTER_PASSES = 10
MASTER_STREAK = 5


def load_doc_mastery_state() -> Dict[str, Dict[str, Any]]:
    try:
        if DOC_MASTERY_FILE.exists():
            data = json.loads(DOC_MASTERY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def save_doc_mastery_state(state: Dict[str, Dict[str, Any]]) -> None:
    try:
        DOC_MASTERY_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def compute_doc_mastery(doc_concepts: List[Dict[str, Any]], tasks_state: Any) -> Dict[str, Dict[str, Any]]:
    """
    Compute mastery information for each concept based on task performance.
    """
    state = _extract_state(tasks_state)
    mastery: Dict[str, Dict[str, Any]] = {}
    for concept in doc_concepts:
        cid = concept.get("id") or concept.get("name")
        if not cid:
            continue
        task_name = task_name_for_concept(concept)
        task_info = state.get(task_name, {})
        passes = int(task_info.get("passes", 0) or 0)
        streak = int(task_info.get("streak", 0) or 0)
        last_status = task_info.get("last_status", "unknown")
        mastered = passes >= MASTER_PASSES and streak >= MASTER_STREAK and last_status == "passing"
        mastery[cid] = {
            "mastered": mastered,
            "passes": passes,
            "streak": streak,
            "last_status": last_status,
            "task": task_name,
        }
    return mastery


def compute_and_save_doc_mastery(
    doc_concepts: List[Dict[str, Any]], tasks_state: Any
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute mastery, persist it, and return any newly mastered concepts.
    """
    previous = load_doc_mastery_state()
    current = compute_doc_mastery(doc_concepts, tasks_state)
    newly_mastered: List[Dict[str, Any]] = []
    for cid, info in current.items():
        prev_info = previous.get(cid, {})
        if not prev_info.get("mastered") and info.get("mastered"):
            newly_mastered.append({"concept": cid, **info})
    save_doc_mastery_state(current)
    return current, newly_mastered
