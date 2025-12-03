from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from manager.neuron_graph import NeuronGraph
from manager.metrics import Metrics


VALUE_STATE_FILE = Path("manager/value_function_state.json")


class ValueFunction:
    """
    Simple value estimator that blends historical rewards with neuron graph context.
    """

    def __init__(self, neuron_graph: NeuronGraph, metrics: Optional[Metrics] = None, alpha: Optional[float] = None) -> None:
        self.neuron_graph = neuron_graph
        self.metrics = metrics
        self.alpha = alpha if alpha is not None else float(os.getenv("VALUE_ALPHA", "0.6"))
        self.state: Dict[str, Dict[str, float]] = {"plugins": {}, "strategies": {}}
        if VALUE_STATE_FILE.exists():
            try:
                loaded = json.loads(VALUE_STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.state.update(loaded)
            except Exception:
                pass
        self._save()

    def _save(self) -> None:
        VALUE_STATE_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _embedding_score(self, node_id: str) -> float:
        node = self.neuron_graph.nodes.get(node_id)
        if not node:
            return 0.0
        embedding = node.get("embedding") or []
        if not embedding:
            return 0.0
        base = sum(float(v) for v in embedding)
        # incorporate neighbors to encourage associative reasoning
        neighbor_bonus = 0.0
        for n in self.neuron_graph.get_neighbors(node_id, top_k=5):
            neighbor_bonus += float(n.get("weight", 0.0)) * 0.1
        return (base / len(embedding)) * 0.1 + neighbor_bonus

    def _historical_score(self, candidate_id: str, candidate_type: str) -> float:
        if candidate_type == "plugin":
            stats = self.state.get("plugins", {}).get(candidate_id, {})
        else:
            stats = self.state.get("strategies", {}).get(candidate_id, {})
        count = float(stats.get("count", 0.0))
        reward_sum = float(stats.get("reward_sum", 0.0))
        if count <= 0:
            return 0.0
        return reward_sum / max(count, 1.0)

    def update_plugin(self, plugin_id: str, reward: float) -> None:
        stats = self.state.setdefault("plugins", {}).setdefault(plugin_id, {"count": 0.0, "reward_sum": 0.0})
        stats["count"] = stats.get("count", 0.0) + 1.0
        stats["reward_sum"] = stats.get("reward_sum", 0.0) + float(reward)
        self.state["plugins"][plugin_id] = stats
        self._save()

    def update_strategy(self, strategy: str, reward: float) -> None:
        stats = self.state.setdefault("strategies", {}).setdefault(strategy, {"count": 0.0, "reward_sum": 0.0})
        stats["count"] = stats.get("count", 0.0) + 1.0
        stats["reward_sum"] = stats.get("reward_sum", 0.0) + float(reward)
        self.state["strategies"][strategy] = stats
        self._save()

    def score(self, candidate_id: str, candidate_type: str = "plugin", strategy: Optional[str] = None) -> float:
        """
        Compute a blended score for a plugin/task/strategy candidate.
        """
        hist = self._historical_score(candidate_id, candidate_type)
        embed_score = self._embedding_score(candidate_id)
        strategy_score = self._historical_score(strategy, "strategy") if strategy else 0.0
        metric_score = 0.0
        if self.metrics and candidate_type == "plugin":
            metric_score = self.metrics.average_reward(candidate_id)
        return hist + self.alpha * embed_score + 0.5 * strategy_score + metric_score
