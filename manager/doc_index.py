from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

Concept = Dict[str, Any]

DOCS_ROOT = Path(os.getenv("PYTHON_DOCS_ROOT", "python-3.14-docs-text"))
FUNCTIONS_FILE = DOCS_ROOT / "library" / "functions.txt"

DOC_FILES = {
    "functions": FUNCTIONS_FILE,
}

SUPPORTED_ENTRIES: List[Dict[str, str]] = [
    {"name": "len", "id": "builtin.len", "section": "builtins", "kind": "function", "file_key": "functions"},
    {"name": "sum", "id": "builtin.sum", "section": "builtins", "kind": "function", "file_key": "functions"},
    {"name": "min", "id": "builtin.min", "section": "builtins", "kind": "function", "file_key": "functions"},
    {"name": "max", "id": "builtin.max", "section": "builtins", "kind": "function", "file_key": "functions"},
    {"name": "abs", "id": "builtin.abs", "section": "builtins", "kind": "function", "file_key": "functions"},
    {"name": "any", "id": "builtin.any", "section": "builtins", "kind": "function", "file_key": "functions"},
    {"name": "all", "id": "builtin.all", "section": "builtins", "kind": "function", "file_key": "functions"},
    {"name": "sorted", "id": "builtin.sorted", "section": "builtins", "kind": "function", "file_key": "functions"},
]


def _read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def _extract_doc(lines: List[str], entry: Dict[str, str]) -> Optional[Concept]:
    """
    Find a function signature and a short snippet that follows it.
    """
    name = entry.get("name", "")
    try:
        pattern = re.compile(f"^{re.escape(name)}\\(")
    except re.error:
        return None
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
        meta = entry
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
    defaults = [
        ("builtin.len", "len", "len(obj)", "Return the number of items in a container."),
        ("builtin.sum", "sum", "sum(iterable, start=0)", "Return the sum of a 'start' value (default: 0) plus an iterable of numbers."),
        ("builtin.min", "min", "min(iterable, *[, key, default])", "Return the smallest item in an iterable or of two or more arguments."),
        ("builtin.max", "max", "max(iterable, *[, key, default])", "Return the largest item in an iterable or of two or more arguments."),
        ("builtin.abs", "abs", "abs(x)", "Return the absolute value of a number."),
        ("builtin.any", "any", "any(iterable)", "Return True if any element of the iterable is true."),
        ("builtin.all", "all", "all(iterable)", "Return True if all elements of the iterable are true."),
        ("builtin.sorted", "sorted", "sorted(iterable)", "Return a new sorted list from the items in iterable."),
    ]
    return [
        {
            "id": cid,
            "kind": "function",
            "name": name,
            "signature": sig,
            "section": "builtins",
            "doc_snippet": snippet,
        }
        for cid, name, sig, snippet in defaults
    ]


def load_concepts() -> List[Concept]:
    """
    Parse a subset of the local Python docs to build concept entries.
    Falls back to a minimal hardcoded list when docs are missing or unreadable.
    """
    line_cache: Dict[str, List[str]] = {}
    for key, path in DOC_FILES.items():
        line_cache[key] = _read_lines(path)
    concepts: List[Concept] = []
    for entry in SUPPORTED_ENTRIES:
        lines = line_cache.get(entry.get("file_key", ""), [])
        if not lines:
            continue
        parsed = _extract_doc(lines, entry)
        if parsed:
            concepts.append(parsed)
    if concepts:
        return concepts
    return _fallback_concepts()
