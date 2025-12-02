from __future__ import annotations

from typing import Any, Dict, List


Concept = Dict[str, Any]


def load_concepts() -> List[Concept]:
    """
    Return a deterministic, minimal list of Python concepts pulled from the
    standard documentation. This initial version is intentionally small and
    hardcoded so the curriculum layer can grow without relying on external I/O.
    """
    concepts: List[Concept] = [
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
        {
            "id": "list.append",
            "kind": "method",
            "name": "list.append",
            "signature": "list.append(x)",
            "section": "list",
            "doc_snippet": "Append object to the end of the list.",
        },
        {
            "id": "dict.get",
            "kind": "method",
            "name": "dict.get",
            "signature": "dict.get(key[, default])",
            "section": "dict",
            "doc_snippet": "Return the value for key if key is in the dictionary, else default.",
        },
    ]
    return concepts
