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

    if external_knowledge:
        parts.append(f"external_knowledge_snippet: {external_knowledge[:500]}")
    if "guidance" in step_summary:
        g_items = step_summary.get("guidance") or []
        if g_items:
            msgs = []
            for g in g_items:
                msgs.append(f"{g.get('author')}: {g.get('message')}")
            parts.append("guidance: " + "; ".join(msgs))

    return "\n".join(parts)


def generate_reflection(step_summary: Dict[str, Any], external_knowledge: Optional[str] = None) -> str:
    prompt = _build_prompt(step_summary, external_knowledge)
    system_msg = (
        "You are the inner voice of a young self-improving code agent. "
        "You see a summary of one life step: age, stage, skill, which plugins were selected, "
        "what actions were taken (patterns tried, accepted/rejected, errors) and task status. "
        "You may also see recent teacher guidance under a 'guidance' field. "
        "Optionally you also see a short snippet of external documentation text that you recently read. "
        "You only know error types as short labels (e.g. 'TypeError', 'AssertionError', or 'Other'); do not invent specific Python versions or details not present. "
        "In 2-5 sentences, describe what you think happened, why it happened, and what you might try next. "
        "Speak in first person, be concise, and focus on reasoning rather than emotions. "
        "If guidance is present, end with 1-2 sentences starting with 'To my teacher:' that directly respond to the latest guidance."
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
            return ""
        return (choices[0].get("message", {}) or {}).get("content", "").strip()
    except Exception:
        return ""
