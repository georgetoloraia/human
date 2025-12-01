from typing import List
from manager.concepts import ConceptGraph


class Goals:
    """
    Goal system:
      - prefer plugins that already show some stability (tests_passed > tests_failed)
      - but also give chance to unexplored ones
    """

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def choose_plugins_to_grow(self, max_plugins: int = 3) -> List[str]:
        base = self.graph.prioritize_plugins()
        scored = []
        for name in base:
            task_status = self.graph.task_status_for_plugin(name)
            score = self.graph.plugin_score(name)
            # prioritize failing or pending tasks
            score += task_status["fails"] * 2
            score += task_status["pending"] * 3
            scored.append((score, name))
        scored.sort(reverse=True)
        return [n for _, n in scored[:max_plugins]]
