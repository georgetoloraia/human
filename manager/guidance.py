from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

GUIDANCE_FILE = Path("manager/guidance_log.json")
DEFAULT_GUIDANCE_HISTORY = 3


def _load_log() -> List[Dict[str, Any]]:
    if not GUIDANCE_FILE.exists():
        return []
    try:
        data = json.loads(GUIDANCE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def append_guidance(author: str, message: str) -> None:
    log = _load_log()
    log.append({"ts": time.time(), "author": author, "message": message})
    GUIDANCE_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")


def latest_guidance(n: int = DEFAULT_GUIDANCE_HISTORY) -> List[Dict[str, Any]]:
    log = _load_log()
    if not log:
        return []
    return log[-n:]
