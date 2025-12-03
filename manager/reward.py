from __future__ import annotations

from typing import Dict, Any


def compute_reward(outcome: Dict[str, Any]) -> float:
    """
    Map an outcome dict to a scalar reward.

    Heuristic:
      +1.0 when tests and evaluation succeed.
      -1.0 on failing tests or unsafe results.
      -0.2 for no-op steps.
    """
    explicit = outcome.get("reward")
    if isinstance(explicit, (int, float)):
        return float(explicit)
    if not outcome:
        return -0.2
    tests_ok = outcome.get("tests_ok")
    eval_ok = outcome.get("eval_ok", tests_ok)
    result = outcome.get("result")
    if result == "unsafe":
        return -1.0
    if tests_ok and eval_ok:
        return 1.0
    if tests_ok is False or eval_ok is False:
        return -1.0
    return -0.2
