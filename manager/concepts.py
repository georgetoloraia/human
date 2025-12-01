from pathlib import Path
import ast
import json
import time

GRAPH_FILE = Path("manager/concept_graph.json")


class ConceptGraph:
    """
    Very simple concept graph:
      - nodes: plugins, functions
      - edges: plugin -> function (contains)
      - attributes: tests_passed, last_modified, growth_count
    """

    def __init__(self):
        self.graph = {
            "_version": "1.0",
            "plugins": {},    # name -> data
            "functions": {},  # plugin.name -> data
            "edges": [],      # (plugin, function)
            "tasks": {},      # task_name -> data
        }
        if GRAPH_FILE.exists():
            try:
                self.graph = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        self.graph.setdefault("_version", "1.0")
        self.graph.setdefault("plugins", {})
        self.graph.setdefault("functions", {})
        self.graph.setdefault("edges", [])
        self.graph.setdefault("tasks", {})

    def _save(self):
        GRAPH_FILE.write_text(json.dumps(self.graph, indent=2), encoding="utf-8")

    def observe_plugin(self, path: Path):
        name = path.name
        src = path.read_text(encoding="utf-8")
        mtime = path.stat().st_mtime
        lines = len(src.splitlines())

        plugin = self.graph["plugins"].get(
            name,
            {
                "name": name,
                "lines": 0,
                "last_modified": 0,
                "growth_count": 0,
                "tests_passed": 0,
                "tests_failed": 0,
            },
        )
        plugin["lines"] = lines
        plugin["last_modified"] = mtime
        self.graph["plugins"][name] = plugin

        try:
            tree = ast.parse(src)
        except Exception:
            self._save()
            return

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fn_key = f"{name}:{node.name}"
                src_segment = ast.get_source_segment(src, node)
                fn_data = self.graph["functions"].get(
                    fn_key,
                    {
                        "plugin": name,
                        "name": node.name,
                        "lines": len(src_segment.splitlines()) if src_segment else 0,
                        "growth_count": 0,
                        "tests_passed": 0,
                        "tests_failed": 0,
                        "last_seen": 0,
                    },
                )
                fn_data["lines"] = fn_data["lines"] or len(node.body)
                fn_data["last_seen"] = time.time()
                self.graph["functions"][fn_key] = fn_data

                edge = (name, fn_key)
                if edge not in self.graph["edges"]:
                    self.graph["edges"].append(edge)

        self._save()

    def record_test_result(self, plugin_name: str, success: bool):
        plugin = self.graph["plugins"].get(plugin_name)
        if plugin:
            if success:
                plugin["tests_passed"] = plugin.get("tests_passed", 0) + 1
            else:
                plugin["tests_failed"] = plugin.get("tests_failed", 0) + 1
            self.graph["plugins"][plugin_name] = plugin
            self._save()

    def record_growth(self, plugin_name: str):
        plugin = self.graph["plugins"].get(plugin_name)
        if plugin:
            plugin["growth_count"] = plugin.get("growth_count", 0) + 1
            self.graph["plugins"][plugin_name] = plugin
            self._save()

    def plugin_score(self, plugin_name: str) -> float:
        p = self.graph["plugins"].get(plugin_name)
        if not p:
            return 0.0
        passed = p.get("tests_passed", 0)
        failed = p.get("tests_failed", 0)
        growth = p.get("growth_count", 0)
        return passed * 2 + growth - failed * 1.5

    def prioritize_plugins(self):
        items = []
        for name in self.graph["plugins"]:
            score = self.plugin_score(name)
            items.append((score, name))
        items.sort(reverse=True)
        return [name for score, name in items]

    def register_tasks(self, tasks):
        for task in tasks:
            name = task.get("name")
            if not name:
                continue
            entry = self.graph["tasks"].get(
                name,
                {
                    "plugin": task.get("target_plugin"),
                    "function": task.get("target_function"),
                    "passed": 0,
                    "failed": 0,
                    "last_success": 0,
                },
            )
            self.graph["tasks"][name] = entry
        self._save()

    def record_task_result(self, task_name: str, success: bool):
        task = self.graph["tasks"].get(task_name)
        if not task:
            return
        if success:
            task["passed"] = task.get("passed", 0) + 1
            task["last_success"] = time.time()
        else:
            task["failed"] = task.get("failed", 0) + 1
        self.graph["tasks"][task_name] = task
        self._save()

    def task_status_for_plugin(self, plugin_name: str):
        fails = 0
        pending = 0
        successes = 0
        for name, task in self.graph["tasks"].items():
            if task.get("plugin") == plugin_name:
                fails += task.get("failed", 0)
                successes += task.get("passed", 0)
                if task.get("passed", 0) == 0:
                    pending += 1
        return {"fails": fails, "pending": pending, "successes": successes}
