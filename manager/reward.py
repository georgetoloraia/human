from __future__ import annotations

from typing import Any, Dict, Tuple

REWARD_CLIP = (-2.0, 2.0)


def _clip(value: float) -> float:
    return max(min(value, REWARD_CLIP[1]), REWARD_CLIP[0])


def compute_reward(outcome: Dict[str, Any], return_details: bool = False) -> float | Tuple[float, Dict[str, Any]]:
    """
    Map a structured outcome dict to a scalar reward.

    Outcome taxonomy (keys are optional):
      - domain: "code" | "env" | other
      - tests_ok / eval_ok: booleans for code runs
      - tests_delta: numeric improvement signal (e.g., passes - fails)
      - regressions: numeric penalty count
      - env_reward: raw environment reward (already scaled)
      - progress: fractional completion progress (0-1)
      - error_type / result: textual hints ("unsafe", "rejected", "noop")

    Reward shaping:
      base reward from success/failure, bonuses for improvements/progress,
      penalties for regressions/unsafe/no-op. Clipped to REWARD_CLIP.
    """
    explicit = outcome.get("reward")
    if isinstance(explicit, (int, float)) and not return_details:
        return _clip(float(explicit))

    domain = outcome.get("domain", "code")
    tests_ok = outcome.get("tests_ok")
    eval_ok = outcome.get("eval_ok", tests_ok)
    tests_delta = float(outcome.get("tests_delta", 0.0) or 0.0)
    regressions = float(outcome.get("regressions", 0.0) or 0.0)
    env_reward = outcome.get("env_reward")
    progress = float(outcome.get("progress", 0.0) or 0.0)
    result = outcome.get("result")
    error_type = outcome.get("error_type")

    base = 0.0
    if tests_ok is True:
        base += 1.0
    elif tests_ok is False:
        base -= 1.0
    if eval_ok is True and tests_ok is not False:
        base += 0.2
    elif eval_ok is False:
        base -= 0.3

    if result == "unsafe":
        base -= 1.5
    elif result in {"rejected", "noop", None}:
        base -= 0.2

    soft_explore = (
        domain == "code"
        and result == "rejected"
        and tests_ok is True
        and regressions <= 0
        and tests_delta >= 0
    )

    improvement_bonus = 0.5 * tests_delta
    regression_penalty = -0.5 * regressions
    env_component = float(env_reward) if isinstance(env_reward, (int, float)) else 0.0
    progress_component = 0.3 * progress

    if soft_explore:
        # exploration without making things worse should be near-neutral
        regression_penalty = 0.0
        progress_component = max(progress_component, 0.2)

    reward = base + improvement_bonus + regression_penalty + env_component + progress_component
    if soft_explore and reward < 0:
        reward = max(reward, -0.05)
    reward = _clip(reward)

    details = {
        "domain": domain,
        "base": base,
        "improvement_bonus": improvement_bonus,
        "regression_penalty": regression_penalty,
        "env_component": env_component,
        "progress_component": progress_component,
        "clipped": reward,
        "soft_explore": soft_explore,
    }

    if isinstance(explicit, (int, float)):
        # explicit reward overrides components but still gets clipped
        reward = _clip(float(explicit))
        details["explicit"] = float(explicit)
        details["clipped"] = reward

    return (reward, details) if return_details else reward
