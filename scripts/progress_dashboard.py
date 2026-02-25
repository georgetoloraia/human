#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(".")
METRICS_PATH = ROOT / "manager/metrics.json"
TASKS_STATE_PATH = ROOT / "manager/tasks_state.json"
DIARY_PATH = ROOT / "manager/mind_diary.json"
COGNITIVE_PATH = ROOT / "manager/cognitive_state.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def extract_rewards(metrics: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    for item in metrics.get("reward_history", []) or []:
        if isinstance(item, (int, float)):
            out.append(float(item))
        elif isinstance(item, dict):
            try:
                out.append(float(item.get("reward", 0.0) or 0.0))
            except Exception:
                out.append(0.0)
    return out


def avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_tasks(tasks_state: Dict[str, Any]) -> Dict[str, Any]:
    total = passing = failing = mastered = 0
    weak: List[Tuple[float, str]] = []
    for name, info in tasks_state.items():
        if str(name).startswith("_") or not isinstance(info, dict):
            continue
        total += 1
        status = info.get("last_status", "unknown")
        streak = int(info.get("streak", 0) or 0)
        passes = int(info.get("passes", 0) or 0)
        fails = int(info.get("fails", 0) or 0)
        if status == "passing":
            passing += 1
        elif status == "failing":
            failing += 1
        if status == "passing" and streak >= 3:
            mastered += 1
        score = streak - (fails * 2) + (passes * 0.1)
        weak.append((score, str(name)))
    weak.sort(key=lambda x: x[0])
    return {
        "total": total,
        "passing": passing,
        "failing": failing,
        "mastered": mastered,
        "weakest": weak[:5],
    }


def diary_step_stats(entry: Dict[str, Any]) -> Dict[str, float]:
    actions = entry.get("actions", []) if isinstance(entry, dict) else []
    accepted = 0
    rejected = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        result = a.get("result")
        if result == "accepted":
            accepted += 1
        elif result in {"rejected", "unsafe"}:
            rejected += 1
    tasks = entry.get("tasks", []) if isinstance(entry, dict) else []
    task_pass_ratio = 0.0
    if isinstance(tasks, list) and tasks:
        known = [t for t in tasks if isinstance(t, dict)]
        if known:
            pass_count = sum(1 for t in known if t.get("last_status") == "passing")
            task_pass_ratio = pass_count / len(known)
    return {
        "accepted": float(accepted),
        "rejected": float(rejected),
        "actions": float(len(actions) if isinstance(actions, list) else 0),
        "task_pass_ratio": task_pass_ratio,
    }


def summarize_diary(diary: List[Dict[str, Any]], window: int) -> Dict[str, Any]:
    if not isinstance(diary, list) or not diary:
        return {}
    valid = [d for d in diary if isinstance(d, dict)]
    if not valid:
        return {}
    last = valid[-1]
    recent = valid[-window:] if window > 0 else valid
    previous = valid[-2 * window : -window] if len(valid) >= 2 * window and window > 0 else []

    def aggregate(entries: List[Dict[str, Any]]) -> Dict[str, float]:
        rows = [diary_step_stats(e) for e in entries]
        if not rows:
            return {"accepted_per_step": 0.0, "rejected_per_step": 0.0, "actions_per_step": 0.0, "task_pass_ratio": 0.0}
        return {
            "accepted_per_step": avg([r["accepted"] for r in rows]),
            "rejected_per_step": avg([r["rejected"] for r in rows]),
            "actions_per_step": avg([r["actions"] for r in rows]),
            "task_pass_ratio": avg([r["task_pass_ratio"] for r in rows]),
        }

    return {
        "count": len(valid),
        "last": last,
        "recent": aggregate(recent),
        "previous": aggregate(previous) if previous else None,
    }


def trend_label(current: float, previous: float | None, eps: float = 1e-6) -> str:
    if previous is None:
        return "n/a"
    diff = current - previous
    if diff > eps:
        return "up"
    if diff < -eps:
        return "down"
    return "flat"


def sparkline(values: List[float], width: int = 24) -> str:
    if not values:
        return ""
    chars = "._-+=*#"
    if len(values) > width:
        step = len(values) / width
        reduced = []
        i = 0.0
        while int(i) < len(values):
            j = min(len(values), int(i + step))
            chunk = values[int(i) : max(int(i) + 1, j)]
            reduced.append(avg(chunk))
            i += step
        values = reduced[:width]
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return chars[len(chars) // 2] * len(values)
    out = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out)


def assess_learning(metrics: Dict[str, Any], tasks: Dict[str, Any], diary_summary: Dict[str, Any], rewards: List[float], window: int) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    steps = int(metrics.get("steps", 0) or 0)
    acc = int(metrics.get("mutations_accepted", 0) or 0)
    rej = int(metrics.get("mutations_rejected", 0) or 0)
    total_mut = acc + rej
    acc_rate = (acc / total_mut) if total_mut else 0.0

    task_summary = summarize_tasks(tasks)
    mastered = task_summary.get("mastered", 0)
    failing = task_summary.get("failing", 0)

    recent_reward = avg(rewards[-window:]) if rewards else 0.0
    prev_reward = avg(rewards[-2 * window : -window]) if len(rewards) >= 2 * window else None
    reward_trend = trend_label(recent_reward, prev_reward)

    diary_recent = (diary_summary or {}).get("recent") or {}
    diary_prev = (diary_summary or {}).get("previous") or {}
    task_pass_ratio = float(diary_recent.get("task_pass_ratio", 0.0) or 0.0)
    task_pass_trend = trend_label(task_pass_ratio, diary_prev.get("task_pass_ratio") if diary_prev else None)
    accepted_per_step = float(diary_recent.get("accepted_per_step", 0.0) or 0.0)

    if steps == 0:
        return "no data", ["no training steps recorded yet"]

    if mastered > 0:
        reasons.append(f"{mastered} tasks mastered")
    if reward_trend == "up":
        reasons.append("recent reward trend is improving")
    elif reward_trend == "down":
        reasons.append("recent reward trend is declining")
    if task_pass_trend == "up":
        reasons.append("task pass ratio is improving")
    elif task_pass_trend == "down":
        reasons.append("task pass ratio is falling")
    if acc_rate > 0:
        reasons.append(f"overall mutation acceptance rate {acc_rate:.1%}")
    if failing > 0:
        reasons.append(f"{failing} tasks currently failing")

    if total_mut >= 20 and acc == 0 and abs(recent_reward) < 1e-9:
        return "stalled", reasons or ["no accepted mutations and zero recent reward"]
    if accepted_per_step <= 0.05 and reward_trend in {"flat", "down"} and failing > 0:
        return "plateau / struggling", reasons
    if mastered > 0 or reward_trend == "up" or task_pass_trend == "up":
        return "learning", reasons
    return "learning slowly", reasons


def print_dashboard(window: int) -> None:
    metrics = load_json(METRICS_PATH, {})
    tasks = load_json(TASKS_STATE_PATH, {})
    diary = load_json(DIARY_PATH, [])
    cognitive = load_json(COGNITIVE_PATH, {})

    if not any([metrics, tasks, diary, cognitive]):
        print("No state files found. Run `python3 -m manager.life_loop` first.")
        return

    rewards = extract_rewards(metrics if isinstance(metrics, dict) else {})
    task_summary = summarize_tasks(tasks if isinstance(tasks, dict) else {})
    diary_summary = summarize_diary(diary if isinstance(diary, list) else [], window)
    verdict, reasons = assess_learning(
        metrics if isinstance(metrics, dict) else {},
        tasks if isinstance(tasks, dict) else {},
        diary_summary,
        rewards,
        window,
    )

    print("=== HUMAN MIND PROGRESS DASHBOARD ===")
    print(f"Learning status: {verdict}")
    if reasons:
        print("Why:", "; ".join(reasons[:4]))
    print()

    if isinstance(metrics, dict):
        steps = int(metrics.get("steps", 0) or 0)
        acc = int(metrics.get("mutations_accepted", 0) or 0)
        rej = int(metrics.get("mutations_rejected", 0) or 0)
        total_mut = acc + rej
        acc_rate = (acc / total_mut) if total_mut else 0.0
        print("Metrics")
        print(f"  steps={steps} accepted={acc} rejected={rej} acceptance_rate={acc_rate:.1%} web_consults={metrics.get('web_consults', 0)}")
        if rewards:
            recent_reward = avg(rewards[-window:])
            prev_reward = avg(rewards[-2 * window : -window]) if len(rewards) >= 2 * window else None
            trend = trend_label(recent_reward, prev_reward)
            prev_text = f"{prev_reward:.3f}" if prev_reward is not None else "n/a"
            print(f"  reward_avg_recent({window})={recent_reward:.3f} prev={prev_text} trend={trend}")
            print(f"  reward_trend: {sparkline(rewards[-max(window * 2, 10):], width=30)}")
        recent_domains = metrics.get("recent_domains", []) or []
        if recent_domains:
            counts: Dict[str, int] = {}
            for d in recent_domains[-50:]:
                counts[str(d)] = counts.get(str(d), 0) + 1
            print(f"  recent_domains(last {min(len(recent_domains),50)}): {counts}")
        print()

    if task_summary:
        total = task_summary.get("total", 0)
        passing = task_summary.get("passing", 0)
        failing = task_summary.get("failing", 0)
        mastered = task_summary.get("mastered", 0)
        pass_ratio = (passing / total) if total else 0.0
        print("Tasks")
        print(f"  total={total} passing={passing} failing={failing} mastered={mastered} pass_ratio={pass_ratio:.1%}")
        weakest = task_summary.get("weakest", [])
        if weakest:
            weak_text = ", ".join(f"{name}({score:.1f})" for score, name in weakest[:5])
            print(f"  weakest: {weak_text}")
        print()

    if diary_summary:
        recent = diary_summary.get("recent") or {}
        prev = diary_summary.get("previous") or {}
        print("Recent Step Behavior")
        print(
            "  "
            f"accepted/step={recent.get('accepted_per_step', 0.0):.2f} "
            f"rejected/step={recent.get('rejected_per_step', 0.0):.2f} "
            f"actions/step={recent.get('actions_per_step', 0.0):.2f} "
            f"task_pass_ratio={recent.get('task_pass_ratio', 0.0):.2%}"
        )
        if prev:
            print(
                "  trends: "
                f"accepted={trend_label(recent.get('accepted_per_step', 0.0), prev.get('accepted_per_step'))}, "
                f"rejected={trend_label(recent.get('rejected_per_step', 0.0), prev.get('rejected_per_step'))}, "
                f"task_pass_ratio={trend_label(recent.get('task_pass_ratio', 0.0), prev.get('task_pass_ratio'))}"
            )
        last = diary_summary.get("last") or {}
        if isinstance(last, dict):
            print(f"  last_step: age={last.get('age')} stage={last.get('stage')} domain={last.get('domain')} phase={last.get('current_phase')}")
            teacher = last.get("teacher_directives") or {}
            if teacher:
                print(
                    "  teacher: "
                    f"style={teacher.get('style')} focus_tasks={teacher.get('focus_tasks', [])} focus_plugins={teacher.get('focus_plugins', [])}"
                )
            cog = last.get("cognitive") or {}
            if cog:
                drives = cog.get("drives", {})
                print(
                    "  cognitive: "
                    f"style={cog.get('style')} "
                    f"curiosity={float(drives.get('curiosity', 0.0)):.2f} "
                    f"caution={float(drives.get('caution', 0.0)):.2f} "
                    f"confidence={float(drives.get('confidence', 0.0)):.2f}"
                )
            reflection = last.get("reflection")
            if isinstance(reflection, str) and reflection.strip():
                print(f"  reflection: {reflection.strip()[:220]}")
        print()

    if isinstance(cognitive, dict) and cognitive:
        drives = cognitive.get("drives", {}) or {}
        print("Cognitive Core")
        print(
            "  "
            f"curiosity={float(drives.get('curiosity', 0.0)):.2f} "
            f"caution={float(drives.get('caution', 0.0)):.2f} "
            f"social={float(drives.get('social', 0.0)):.2f} "
            f"confidence={float(drives.get('confidence', 0.0)):.2f}"
        )
        print(f"  intentions={cognitive.get('intentions', [])}")
        print(f"  beliefs={cognitive.get('beliefs', {})}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Show whether the local self-learning mind is improving.")
    ap.add_argument("--window", type=int, default=20, help="Recent window size for trend comparisons.")
    args = ap.parse_args()
    print_dashboard(window=max(1, args.window))


if __name__ == "__main__":
    main()
