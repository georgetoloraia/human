from __future__ import annotations

from typing import Any, Dict, List, Optional

from .doc_index import Concept

TASK_PREFIX = "doc_"
DEFAULT_REQUIRED_STREAK = 3
DEFAULT_MIN_PASSES = 3


def _extract_state(tasks_state: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize TaskStateManager-like objects or plain dicts into a simple mapping.
    """
    if hasattr(tasks_state, "state"):
        return getattr(tasks_state, "state") or {}
    if isinstance(tasks_state, dict):
        return tasks_state
    return {}


def task_name_for_concept(concept: Concept) -> str:
    """
    Map a concept to the canonical task name we expect to generate.
    """
    raw = concept.get("name") or concept.get("id") or "concept"
    safe = raw.replace(".", "_")
    return f"{TASK_PREFIX}{safe}"


def _is_learned(task_info: Optional[Dict[str, Any]]) -> bool:
    if not task_info:
        return False
    passes = int(task_info.get("passes", 0) or 0)
    streak = int(task_info.get("streak", 0) or 0)
    status = task_info.get("last_status")
    if streak >= DEFAULT_REQUIRED_STREAK and status == "passing":
        return True
    if passes >= DEFAULT_MIN_PASSES and status == "passing":
        return True
    return False


def choose_next_concept(tasks_state: Any, doc_concepts: List[Concept]) -> Optional[Concept]:
    """
    Given the current task performance state, pick the next unlearned concept
    in document order. Returns None if everything in the index looks learned.
    """
    state = _extract_state(tasks_state)
    for concept in doc_concepts:
        task_name = task_name_for_concept(concept)
        info = state.get(task_name)
        if _is_learned(info):
            continue
        return concept
    return None
