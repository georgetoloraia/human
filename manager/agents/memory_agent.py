from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from manager.agents import BaseAgent

if TYPE_CHECKING:  # pragma: no cover
    from manager.mind import Mind


class MemoryAgent(BaseAgent):
    """
    Persists outcomes to long-term stores (diary, neuron graph, metrics).
    """

    def __init__(self, mind: "Mind") -> None:
        super().__init__()
        self.mind = mind

    def act(self) -> Dict[str, object]:
        # flush neuron graph snapshot; diary is handled by the main loop
        self.mind.neuron_graph.save()
        return {"status": "saved"}
