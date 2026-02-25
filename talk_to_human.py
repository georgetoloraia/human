#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from manager.guidance import append_guidance, latest_guidance
from manager.cognitive_core import CognitiveCore

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


DIARY_PATH = Path("manager/mind_diary.json")
MEMORY_PATH = Path("manager/brain_memory.json")
LLM_BASE_URL = os.getenv("MIND_LLM_BASE_URL", "http://127.0.0.1:11434/v1/chat/completions")
LLM_MODEL = os.getenv("MIND_LLM_MODEL", "llama3.2:1b")
LLM_TIMEOUT = float(os.getenv("MIND_LLM_TIMEOUT", "60"))


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return default


def _latest_reflection() -> str | None:
    data = _load_json(DIARY_PATH, [])
    if not isinstance(data, list) or not data:
        return None
    last = data[-1] if isinstance(data[-1], dict) else {}
    reflection = last.get("reflection")
    if isinstance(reflection, str) and reflection.strip():
        return reflection.strip()
    return None


def _mind_status() -> Dict[str, Any]:
    data = _load_json(MEMORY_PATH, {})
    if not isinstance(data, dict):
        return {}
    attempts = data.get("attempts", {}) or {}
    ok_total = 0
    for v in attempts.values():
        if isinstance(v, dict):
            ok_total += int(v.get("ok", 0) or 0)
    return {
        "age": int(data.get("age", 0) or 0),
        "skill": int(data.get("age", 0) or 0) + ok_total,
        "meta_skill": float(data.get("meta_skill", 0.0) or 0.0),
    }


def _build_llm_messages(user_message: str) -> List[Dict[str, str]]:
    guidance = latest_guidance(5)
    reflection = _latest_reflection()
    status = _mind_status()
    guidance_lines = []
    for item in guidance:
        if isinstance(item, dict):
            guidance_lines.append(f"- {item.get('author', 'user')}: {item.get('message', '')}")
    context = [
        f"Mind status: age={status.get('age', 0)}, skill={status.get('skill', 0)}, meta_skill={status.get('meta_skill', 0.0):.2f}",
        "Recent guidance:",
        *guidance_lines[-5:],
    ]
    if reflection:
        context.append(f"Latest internal reflection: {reflection[:600]}")

    system = (
        "You are a learning AI prototype speaking to your human teacher. "
        "Be honest, simple, and conversational. "
        "Do not claim human consciousness. "
        "Acknowledge what you can do now (learn tasks, accept guidance, improve code) and what you cannot do yet."
    )
    user = f"Context:\n" + "\n".join(context) + f"\n\nTeacher message: {user_message}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _llm_reply(user_message: str) -> str | None:
    if requests is None:
        return None
    body = {
        "model": LLM_MODEL,
        "messages": _build_llm_messages(user_message),
        "temperature": 0.4,
        "stream": False,
    }
    try:
        resp = requests.post(LLM_BASE_URL, json=body, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        content = ((choices[0] or {}).get("message") or {}).get("content", "")
        content = str(content).strip()
        return content or None
    except Exception:
        return None


def _fallback_reply(user_message: str) -> str:
    status = _mind_status()
    reflection = _latest_reflection()
    msg = user_message.strip().lower()
    if any(k in msg for k in ["who are you", "what are you"]):
        return (
            "I am a learning code-agent prototype in this repo. "
            "I can store your guidance, run a self-improving loop, and report progress."
        )
    if any(k in msg for k in ["learn", "improve", "grow"]):
        return (
            f"I am learning step by step. Current state: age={status.get('age', 0)}, "
            f"skill={status.get('skill', 0)}, meta_skill={status.get('meta_skill', 0.0):.2f}. "
            "Run `python3 -m manager.life_loop` to keep training me."
        )
    if reflection:
        return f"I recorded your message and will use it in my next steps. Latest reflection: {reflection}"
    return "I recorded your message. Run `python3 -m manager.life_loop` so I can learn from tasks and update my diary."


def _external_state_for_cognition() -> Dict[str, Any]:
    status = _mind_status()
    return {
        **status,
        "last_reflection": _latest_reflection(),
    }


def respond_to_teacher(message: str, use_llm: bool = True) -> str:
    append_guidance(author="human_teacher", message=message)
    if use_llm:
        reply = _llm_reply(message)
        if reply:
            return reply
    try:
        core = CognitiveCore()
        return core.respond(message, external=_external_state_for_cognition())
    except Exception:
        return _fallback_reply(message)


def run_interactive(use_llm: bool = True) -> int:
    print("Human chat mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            return 0
        reply = respond_to_teacher(message, use_llm=use_llm)
        print(f"mind> {reply}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Talk to the learning mind (stores guidance and optionally replies).")
    ap.add_argument("message", nargs="?", help="One-shot message to the mind.")
    ap.add_argument("-i", "--interactive", action="store_true", help="Start interactive chat mode.")
    ap.add_argument("--no-llm", action="store_true", help="Disable LLM reply and use local fallback response.")
    args = ap.parse_args()

    if args.interactive or not args.message:
        return run_interactive(use_llm=not args.no_llm)

    reply = respond_to_teacher(args.message, use_llm=not args.no_llm)
    print(f"Recorded guidance: {args.message}")
    print(f"Mind reply: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
