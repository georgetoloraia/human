import os
from pathlib import Path
from typing import List, Dict

TASKS_DIR = Path("tasks")
GENERATED_DIR = Path("tests/generated")


def _parse_task_file(path: Path) -> Dict:
    data = {"requirements": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped and not stripped.startswith("-"):
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "requirements":
                data.setdefault("requirements", [])
            else:
                data[key] = val
        elif stripped.startswith("-"):
            req = stripped.lstrip("-").strip()
            if req:
                data.setdefault("requirements", []).append(req)
    return data


def load_tasks() -> List[Dict]:
    tasks: List[Dict] = []
    if not TASKS_DIR.exists():
        return tasks
    for path in TASKS_DIR.glob("*.yml"):
        task = _parse_task_file(path)
        if "name" in task and "target_plugin" in task and "target_function" in task:
            tasks.append(task)
    return tasks


def _task_to_test_source(task: Dict) -> str:
    name = task["name"]
    plugin = task["target_plugin"].replace(".py", "")
    func = task["target_function"]
    reqs = task.get("requirements", [])
    lines = [
        "import sys",
        "from pathlib import Path",
        "",
        "ROOT = Path(__file__).resolve().parents[2]",
        "if str(ROOT) not in sys.path:",
        "    sys.path.insert(0, str(ROOT))",
        f"from plugins.{plugin} import {func}",
        "",
    ]
    for idx, req in enumerate(reqs, 1):
        lines.append(f"def test_{name}_{idx}():")
        lines.append(f"    assert {req}")
        lines.append("")
    return "\n".join(lines)


def regenerate_task_tests() -> List[Dict]:
    tasks = load_tasks()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    # clear existing generated tests
    for old in GENERATED_DIR.glob("test_*.py"):
        old.unlink()
    for task in tasks:
        test_src = _task_to_test_source(task)
        out_path = GENERATED_DIR / f"test_{task['name']}.py"
        out_path.write_text(test_src, encoding="utf-8")
    return tasks


if __name__ == "__main__":
    regenerate_task_tests()
