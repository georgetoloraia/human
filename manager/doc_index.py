from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

Concept = Dict[str, Any]

DOCS_ROOT = Path(os.getenv("PYTHON_DOCS_ROOT", "python-3.14-docs-text"))
FUNCTIONS_FILE = DOCS_ROOT / "library" / "functions.txt"

SUPPORTED_FUNCTIONS = {
    "len": {
        "id": "builtin.len",
        "section": "builtins",
    },
    "sum": {
        "id": "builtin.sum",
        "section": "builtins",
    },
    "min": {
        "id": "builtin.min",
        "section": "builtins",
    },
    "max": {
        "id": "builtin.max",
        "section": "builtins",
    },
}


def _read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def _extract_doc(lines: List[str], name: str) -> Optional[Concept]:
    """
    Find a function signature and a short snippet that follows it.
    """
    pattern = re.compile(rf"^{re.escape(name)}\\(")
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not pattern.match(line):
            continue
        signature = line
        # skip blank lines immediately following the signature
        i = idx + 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        snippet_lines: List[str] = []
        while i < len(lines):
            candidate = lines[i]
            stripped = candidate.strip()
            if not stripped:
                if snippet_lines:
                    break
                i += 1
                continue
            if not candidate.startswith(" "):
                # likely the start of the next definition
                break
            snippet_lines.append(stripped)
            if len(snippet_lines) >= 3:
                break
            i += 1
        doc_snippet = " ".join(snippet_lines).strip()
        meta = SUPPORTED_FUNCTIONS.get(name, {})
        if not doc_snippet:
            continue
        return {
            "id": meta.get("id", f"builtin.{name}"),
            "kind": "function",
            "name": name,
            "signature": signature,
            "section": meta.get("section", "builtins"),
            "doc_snippet": doc_snippet,
        }
    return None


def _fallback_concepts() -> List[Concept]:
    return [
        {
            "id": "builtin.len",
            "kind": "function",
            "name": "len",
            "signature": "len(obj)",
            "section": "builtins",
            "doc_snippet": "Return the number of items in a container.",
        },
        {
            "id": "builtin.sum",
            "kind": "function",
            "name": "sum",
            "signature": "sum(iterable, start=0)",
            "section": "builtins",
            "doc_snippet": "Return the sum of a 'start' value (default: 0) plus an iterable of numbers.",
        },
        {
            "id": "builtin.min",
            "kind": "function",
            "name": "min",
            "signature": "min(iterable, *[, key, default])",
            "section": "builtins",
            "doc_snippet": "Return the smallest item in an iterable or of two or more arguments.",
        },
        {
            "id": "builtin.max",
            "kind": "function",
            "name": "max",
            "signature": "max(iterable, *[, key, default])",
            "section": "builtins",
            "doc_snippet": "Return the largest item in an iterable or of two or more arguments.",
        },
    ]


def load_concepts() -> List[Concept]:
    """
    Parse a subset of the local Python docs to build concept entries.
    Falls back to a minimal hardcoded list when docs are missing or unreadable.
    """
    lines = _read_lines(FUNCTIONS_FILE)
    concepts: List[Concept] = []
    for name in SUPPORTED_FUNCTIONS.keys():
        if not lines:
            break
        parsed = _extract_doc(lines, name)
        if parsed:
            concepts.append(parsed)
    if concepts:
        return concepts
    return _fallback_concepts()
