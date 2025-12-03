from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .concept_miner import task_name_for_concept

TASKS_DIR = Path("tasks")
PLUGIN_PATH = Path("plugins/sample_plugin.py")


def _task_to_yaml(spec: Dict[str, Any]) -> str:
    lines = [
        f"name: {spec['name']}",
        f"phase: {spec.get('phase', 1)}",
        f"difficulty: {spec.get('difficulty', 1)}",
        f"target_plugin: {spec.get('target_plugin', 'sample_plugin.py')}",
        f"target_function: {spec['target_function']}",
        f"description: {spec.get('description', '')}",
        "requirements:",
    ]
    for req in spec.get("requirements", []):
        lines.append(f"  - {req}")
    return "\n".join(lines) + "\n"


def _task_spec_for_concept(concept: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map a concept to a concrete task and stub implementation.
    Returns None when the concept is not supported yet.
    """
    name = concept.get("name") or concept.get("id") or ""
    task_name = task_name_for_concept(concept)

    if name in {"len", "builtin.len"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_len",
            "description": "Use Python's built-in len() to compute the length of a sequence.",
            "requirements": [
                "use_len([1,2,3]) == 3",
                "use_len('hello') == 5",
                "use_len([]) == 0",
            ],
            "impl": [
                "def use_len(obj):",
                "    return len(obj)",
            ],
        }

    if name in {"sum", "builtin.sum"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_sum",
            "description": "Use Python's built-in sum() to total an iterable of numbers.",
            "requirements": [
                "use_sum([1,2,3]) == 6",
                "use_sum([]) == 0",
                "use_sum([-2,2,5]) == 5",
            ],
            "impl": [
                "def use_sum(values):",
                "    return sum(values)",
            ],
        }

    if name in {"min", "builtin.min"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_min",
            "description": "Use Python's built-in min() to find the smallest value.",
            "requirements": [
                "use_min([3,1,2]) == 1",
                "use_min([-5,-2,10]) == -5",
                "use_min([7]) == 7",
            ],
            "impl": [
                "def use_min(values):",
                "    return min(values)",
            ],
        }

    if name in {"max", "builtin.max"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_max",
            "description": "Use Python's built-in max() to find the largest value.",
            "requirements": [
                "use_max([3,1,2]) == 3",
                "use_max([-5,-2,10]) == 10",
                "use_max([7]) == 7",
            ],
            "impl": [
                "def use_max(values):",
                "    return max(values)",
            ],
        }

    if name in {"abs", "builtin.abs"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_abs",
            "description": "Use Python's built-in abs() to return the absolute value.",
            "requirements": [
                "use_abs(-3) == 3",
                "use_abs(0) == 0",
                "use_abs(7) == 7",
            ],
            "impl": [
                "def use_abs(value):",
                "    return abs(value)",
            ],
        }

    if name in {"any", "builtin.any"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_any",
            "description": "Use Python's built-in any() to test truthiness in an iterable.",
            "requirements": [
                "use_any([False, False, True]) is True",
                "use_any([]) is False",
                "use_any([0, '', []]) is False",
            ],
            "impl": [
                "def use_any(values):",
                "    return any(values)",
            ],
        }

    if name in {"all", "builtin.all"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_all",
            "description": "Use Python's built-in all() to ensure every element is truthy.",
            "requirements": [
                "use_all([True, True]) is True",
                "use_all([True, False]) is False",
                "use_all([]) is True",
            ],
            "impl": [
                "def use_all(values):",
                "    return all(values)",
            ],
        }

    if name in {"sorted", "builtin.sorted"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_sorted",
            "description": "Use Python's built-in sorted() to return a sorted list from an iterable.",
            "requirements": [
                "use_sorted([3,1,2]) == [1,2,3]",
                "use_sorted(['b','a']) == ['a','b']",
                "use_sorted([]) == []",
            ],
            "impl": [
                "def use_sorted(values):",
                "    return sorted(values)",
            ],
        }

    if name in {"list.append", "list_append"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_list_append",
            "description": "Practice list.append by adding an element to the end of a list copy.",
            "requirements": [
                "use_list_append([1,2], 3) == [1,2,3]",
                "use_list_append([], 'a') == ['a']",
                "use_list_append(['x'], 'y') == ['x','y']",
            ],
            "impl": [
                "def use_list_append(values, item):",
                "    result = list(values)",
                "    result.append(item)",
                "    return result",
            ],
        }

    if name in {"dict.get", "dict_get"}:
        return {
            "name": task_name,
            "phase": 1,
            "difficulty": 1,
            "target_plugin": "sample_plugin.py",
            "target_function": "use_dict_get",
            "description": "Use dict.get to safely retrieve values with a default when missing.",
            "requirements": [
                "use_dict_get({'a': 1}, 'a', 0) == 1",
                "use_dict_get({'a': 1}, 'b', 0) == 0",
                "use_dict_get({}, 'missing', 'fallback') == 'fallback'",
            ],
            "impl": [
                "def use_dict_get(mapping, key, default=None):",
                "    data = mapping or {}",
                "    return data.get(key, default)",
            ],
        }

    return None


def _ensure_plugin_function(func_name: str, impl_lines: List[str]) -> bool:
    if not PLUGIN_PATH.exists():
        return False
    src = PLUGIN_PATH.read_text(encoding="utf-8")
    if f"def {func_name}(" in src:
        return False
    snippet = "\n".join(impl_lines)
    with PLUGIN_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n" + snippet + "\n")
    return True


def ensure_tasks_for_concept(concept: Dict[str, Any]) -> List[str]:
    """
    Create YAML tasks and plugin stubs for the given concept if they are missing.
    Returns a list of created task file paths.
    """
    spec = _task_spec_for_concept(concept)
    if not spec:
        return []
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    task_path = TASKS_DIR / f"{spec['name']}.yml"
    if not task_path.exists():
        task_yaml = _task_to_yaml(spec)
        task_path.write_text(task_yaml, encoding="utf-8")
        created.append(str(task_path))

    _ensure_plugin_function(spec["target_function"], spec.get("impl", []))
    return created
