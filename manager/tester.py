from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from manager.tasks import Task, load_tasks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ERROR = "Other"


def _plugin_module_name(plugin_path: str) -> str:
    """
    Convert a plugin file name like 'sample_plugin.py' to an importable module name.
    """
    return Path(plugin_path).stem


def classify_error(exc: Exception | None) -> str:
    if exc is None:
        return DEFAULT_ERROR
    name = exc.__class__.__name__ or DEFAULT_ERROR
    return name


def _build_eval_env(module, target_function: str | None) -> Dict[str, object]:
    """
    Build the globals used for evaluating task requirements.
    """
    env: Dict[str, object] = {"__builtins__": __builtins__}
    env.update({k: v for k, v in module.__dict__.items() if not k.startswith("_")})
    if target_function:
        env[target_function] = getattr(module, target_function)
    return env


def _evaluate_requirement(expr: str, env: Dict[str, object]) -> Tuple[bool, str]:
    try:
        result = eval(expr, env, {})
        if result is True:
            return True, "OK"
        return False, "AssertionError"
    except AssertionError:
        return False, "AssertionError"
    except Exception as exc:  # pylint: disable=broad-except
        return False, classify_error(exc)


def _run_single_task(task: Task) -> Tuple[bool, str]:
    try:
        module_name = _plugin_module_name(task.target_plugin)
        module = importlib.import_module(f"plugins.{module_name}")
    except Exception as exc:  # pylint: disable=broad-except
        return False, classify_error(exc)

    try:
        env = _build_eval_env(module, task.target_function)
    except Exception as exc:  # pylint: disable=broad-except
        return False, classify_error(exc)

    if not task.requirements:
        return True, "OK"

    for req in task.requirements:
        ok, err_type = _evaluate_requirement(req, env)
        if not ok:
            return False, err_type
    return True, "OK"


def run_tests(plugin_name: str | None = None) -> tuple[bool, str, List[str], Dict[str, Tuple[bool, str]]]:
    """
    Evaluate YAML-defined tasks against their target plugins.

    Returns:
        (all_passed, error_type, failing_tasks, task_results)
        - all_passed: True only if every evaluated task passes.
        - error_type: "OK" if all passed, otherwise the first failing error type.
        - failing_tasks: list of task names that failed.
        - task_results: mapping of task name -> (passed, error_type).
    """
    tasks = load_tasks()
    if plugin_name:
        tasks = {name: t for name, t in tasks.items() if t.target_plugin == plugin_name}

    failing: List[str] = []
    results: Dict[str, Tuple[bool, str]] = {}

    if not tasks:
        return True, "OK", failing, results

    overall_error = "OK"
    all_passed = True
    for tname, task in tasks.items():
        passed, err_type = _run_single_task(task)
        results[tname] = (passed, err_type)
        if not passed:
            failing.append(tname)
            all_passed = False
            if overall_error == "OK":
                overall_error = err_type

    if all_passed:
        overall_error = "OK"
    elif overall_error == "OK":
        overall_error = results[failing[0]][1] if failing else DEFAULT_ERROR

    return all_passed, overall_error, failing, results
