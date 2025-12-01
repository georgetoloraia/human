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
        ordered = self.graph.prioritize_plugins()
        if not ordered:
            return []
        top = ordered[:max_plugins]
        return top
