from typing import List
from manager.concepts import ConceptGraph
from manager.curriculum import Curriculum


class Goals:
    """
    Goal system:
      - prefer plugins that already show some stability (tests_passed > tests_failed)
      - but also give chance to unexplored ones
    """

    def __init__(self, graph: ConceptGraph, curriculum: Curriculum):
        self.graph = graph
        self.curriculum = curriculum
        self.focus_queue: List[str] = []

    def choose_plugins_to_grow(self, max_plugins: int = 3) -> List[str]:
        if self.focus_queue:
            return [p for p in self.focus_queue[:max_plugins]]

        base = self.graph.prioritize_plugins()
        scored = []
        for name in base:
            task_status = self.graph.task_status_for_plugin(name)
            score = self.graph.plugin_score(name)
            # prioritize failing or pending tasks
            score += task_status["fails"] * 2
            score += task_status["pending"] * 3
            for tname in self.curriculum.tasks_by_plugin().get(name, []):
                tinfo = self.curriculum.get_task_info(tname)
                if tinfo.get("status") == "unlocked":
                    score += 5
                if tinfo.get("phase") == self.curriculum.state.get("current_phase"):
                    score += 3
            scored.append((score, name))
        scored.sort(reverse=True)
        return [n for _, n in scored[:max_plugins]]

    def force_focus(self, task_name):
        if not task_name:
            return
        # map task to plugin via curriculum
        task_info = self.curriculum.get_task_info(task_name)
        plugin = task_info.get("plugin")
        if not plugin:
            return
        if plugin in self.focus_queue:
            return
        self.focus_queue.insert(0, plugin)
