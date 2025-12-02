from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

LLM_BASE_URL = os.getenv("MIND_LLM_BASE_URL", "http://127.0.0.1:11434/v1/chat/completions")
LLM_MODEL = os.getenv("MIND_LLM_MODEL", "llama3.2:1b")
LLM_TIMEOUT = float(os.getenv("MIND_LLM_TIMEOUT", "300"))


def _build_prompt(step_summary: Dict[str, Any], external_knowledge: Optional[str]) -> str:
    age = step_summary.get("age")
    stage = step_summary.get("stage")
    skill = step_summary.get("skill")
    selected_plugins = step_summary.get("selected_plugins", [])
    actions = step_summary.get("actions", [])
    tasks = step_summary.get("tasks", [])

    parts = [f"age={age}, stage={stage}, skill={skill}"]

    if selected_plugins:
        plugin_lines = []
        for sel in selected_plugins:
            plugin_lines.append(
                f"{sel.get('plugin')} "
                f"(score={sel.get('total_score')}, "
                f"task_failing={sel.get('task_failing_count')}, "
                f"task_streak={sel.get('task_avg_streak')})"
            )
        parts.append("selected_plugins: " + "; ".join(plugin_lines))

    if actions:
        action_lines = []
        for a in actions:
            line = (
                f"{a.get('plugin')} "
                f"pattern={a.get('pattern')} "
                f"result={a.get('result')} "
                f"error_type={a.get('error_type')}"
            )
            if "web_consult" in a:
                wc = a["web_consult"]
                line += f" web_consult(url={wc.get('url')}, status={wc.get('status')})"
            action_lines.append(line)
        parts.append("actions: " + "; ".join(action_lines))

    if tasks:
        task_lines = []
        for t in tasks:
            task_lines.append(
                f"{t.get('name')}@{t.get('plugin')} "
                f"streak={t.get('streak')} "
                f"passes={t.get('passes')} "
                f"fails={t.get('fails')} "
                f"status={t.get('last_status')} "
                f"err={t.get('last_error_type')}"
            )
        parts.append("tasks: " + "; ".join(task_lines))

    doc_events = step_summary.get("doc_curriculum") or []
    if isinstance(doc_events, dict):
        doc_events = [doc_events]
    if doc_events:
        doc_lines = []
        for ev in doc_events:
            bits = [ev.get("action", "doc_event"), str(ev.get("concept") or "unknown_concept")]
            task_files = ev.get("task_files") or ev.get("tasks") or []
            if task_files:
                bits.append("tasks=" + ",".join(task_files))
            snippet = ev.get("doc_snippet")
            if snippet:
                bits.append(f"note={snippet}")
            passes = ev.get("passes")
            streak = ev.get("streak")
            if passes is not None:
                bits.append(f"passes={passes}")
            if streak is not None:
                bits.append(f"streak={streak}")
            doc_lines.append(" ".join(str(b) for b in bits if b))
        parts.append("doc_curriculum: " + "; ".join(doc_lines))

    if external_knowledge:
        parts.append(f"external_knowledge_snippet: {external_knowledge[:500]}")
    guidance_items = step_summary.get("guidance") or []
    last_guidance = step_summary.get("last_guidance")
    if guidance_items:
        parts.append("guidance_messages:")
        for g in guidance_items:
            parts.append(f"- {g.get('author')}: {g.get('message')}")
    if last_guidance:
        parts.append(f"latest_guidance: {last_guidance.get('author')}: {last_guidance.get('message')}")

    return "\n".join(parts)


def _fallback_reflection(step_summary: Dict[str, Any]) -> str:
    age = step_summary.get("age")
    stage = step_summary.get("stage")
    skill = step_summary.get("skill")
    actions = step_summary.get("actions") or []
    tasks = step_summary.get("tasks") or []
    doc_events = step_summary.get("doc_curriculum")
    if isinstance(doc_events, dict):
        doc_events = [doc_events]

    passing = sum(1 for t in tasks if t.get("last_status") == "passing")
    failing = sum(1 for t in tasks if t.get("last_status") == "failing")
    task_note = f"{passing} passing, {failing} failing" if tasks else "no tasks"

    action_note = "none"
    if actions:
        a = actions[-1]
        action_note = (
            f"{a.get('plugin')} pattern={a.get('pattern')} result={a.get('result')} error={a.get('error_type')}"
        )

    doc_note = ""
    if doc_events:
        recent = doc_events[-1]
        doc_note = f" New concept focus: {recent.get('concept')} ({recent.get('action')})."

    part_one = f"Now: age {age}, stage {stage}, skill {skill}; tasks {task_note}; recent action {action_note}.{doc_note}"
    last_guidance = step_summary.get("last_guidance") or {}
    guidance_msg = last_guidance.get("message")
    if guidance_msg:
        part_two = f"To my teacher: I see your latest guidance '{guidance_msg}'. I'm working on the current tasks and will keep reporting progress."
    else:
        part_two = "To my teacher: No new guidance yet, but I'm continuing with the current practice set."
    return f"{part_one} {part_two}"


def generate_reflection(step_summary: Dict[str, Any], external_knowledge: Optional[str] = None) -> str:
    prompt = _build_prompt(step_summary, external_knowledge)
    system_msg = (
        "You are the inner voice of a young self-improving code agent. "
        "You see a summary of one life step: age, stage, skill, which plugins were selected, "
        "what actions were taken (patterns tried, accepted/rejected, errors), task status, "
        "and optionally doc_curriculum entries describing new concepts and tasks derived from Python docs, "
        "and optionally external knowledge snippets or guidance messages from a human teacher. "
        "You only know error types as short labels (e.g. 'TypeError', 'AssertionError', or 'Other'); do not invent details not present. "
        "Always respond in two parts, in this order: "
        "1) briefly describe what you are doing now based on this step (actions, tasks, successes/failures, and if present the newest doc concept/tasks) using the provided age/stage/skill as-is; "
        "2) then write 1–2 sentences starting with 'To my teacher:' that directly respond to the MOST RECENT guidance message (or note that none was given), in your own words. "
        "Be concise and honest about uncertainty. If you see doc_curriculum details (added_concept or mastered_concept), mention what new concept is being added or practiced and whether you feel ready to move on."
    )
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "stream": False,
    }
    try:
        resp = requests.post(LLM_BASE_URL, json=body, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return _fallback_reflection(step_summary)
        content = (choices[0].get("message", {}) or {}).get("content", "").strip()
        return content or _fallback_reflection(step_summary)
    except Exception:
        return _fallback_reflection(step_summary)
