from pathlib import Path
import ast
import copy
from random import choice, random

SAFE_GROW_LINES = [
    "pass",
    "x = x",
    "return x",
    "x = x + 0",
    "x = int(x)",
]


def _module_to_source(tree: ast.Module) -> str:
    parts = []
    for node in tree.body:
        parts.append(ast.unparse(node))
    return "\n\n".join(parts) + "\n"


def grow_function(fn: ast.FunctionDef) -> ast.FunctionDef:
    new_fn = copy.deepcopy(fn)
    new_stmt = ast.parse(choice(SAFE_GROW_LINES)).body[0]
    # Insert before final return if present, otherwise near the end.
    insert_at = len(new_fn.body) - 1 if new_fn.body else 0
    new_fn.body.insert(insert_at, new_stmt)
    ast.fix_missing_locations(new_fn)
    return new_fn


def grow_code(src: str) -> str | None:
    tree = ast.parse(src)
    mutated = False

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and random() < 0.5:
            tree.body[i] = grow_function(node)
            mutated = True

    return _module_to_source(tree) if mutated else None
