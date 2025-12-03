from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_GRAPH_PATH = Path("manager/neuron_graph.json")
GRAPH_VERSION = "0.1"


class NeuronGraph:
    """
    Lightweight embedding-backed graph for relating plugins, functions, errors, tasks, and reflections.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_GRAPH_PATH
        self.graph: Dict[str, object] = {"version": GRAPH_VERSION, "nodes": {}, "edges": []}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.graph.update(loaded)
            except Exception:
                # corrupted files fall back to a fresh graph
                self.graph = {"version": GRAPH_VERSION, "nodes": {}, "edges": []}
        self.graph.setdefault("version", GRAPH_VERSION)
        self.graph.setdefault("nodes", {})
        self.graph.setdefault("edges", [])

    @property
    def nodes(self) -> Dict[str, Dict[str, object]]:
        return self.graph.setdefault("nodes", {})  # type: ignore[return-value]

    @property
    def edges(self) -> List[Dict[str, object]]:
        return self.graph.setdefault("edges", [])  # type: ignore[return-value]

    def add_node(self, node_id: str, node_type: str, embedding: List[float], metadata: Optional[Dict[str, object]] = None) -> None:
        meta = metadata or {}
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "embedding": list(embedding),
            "meta": meta,
        }
        self.save()

    def add_edge(self, source_id: str, target_id: str, weight: float = 0.0, meta: Optional[Dict[str, object]] = None) -> None:
        meta = meta or {}
        existing = next((e for e in self.edges if e.get("source") == source_id and e.get("target") == target_id), None)
        if existing:
            existing["weight"] = weight
            existing_meta = existing.get("meta") or {}
            existing_meta.update(meta)
            existing["meta"] = existing_meta
        else:
            self.edges.append({"source": source_id, "target": target_id, "weight": float(weight), "meta": meta})
        self.save()

    def update_edge_weight(self, source_id: str, target_id: str, delta: float) -> None:
        existing = next((e for e in self.edges if e.get("source") == source_id and e.get("target") == target_id), None)
        if not existing:
            self.add_edge(source_id, target_id, weight=delta)
            return
        existing["weight"] = float(existing.get("weight", 0.0)) + float(delta)
        # simple clipping to avoid runaway weights
        existing["weight"] = max(min(existing["weight"], 10.0), -10.0)
        self.save()

    def get_neighbors(self, node_id: str, top_k: int = 10) -> List[Dict[str, object]]:
        neighbors: List[Dict[str, object]] = []
        for edge in self.edges:
            if edge.get("source") == node_id:
                other_id = edge.get("target")
            elif edge.get("target") == node_id:
                other_id = edge.get("source")
            else:
                continue
            other = self.nodes.get(str(other_id), {"id": other_id, "type": "unknown", "embedding": [], "meta": {}})
            neighbors.append(
                {
                    "id": other_id,
                    "type": other.get("type"),
                    "weight": edge.get("weight", 0.0),
                    "meta": other.get("meta", {}),
                }
            )
        neighbors.sort(key=lambda n: n.get("weight", 0.0), reverse=True)
        return neighbors[:top_k]

    def save(self, path: Path | None = None) -> None:
        out_path = path or self.path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.graph, indent=2), encoding="utf-8")

    def load(self, path: Path | None = None) -> None:
        src = path or self.path
        if not src.exists():
            return
        loaded = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return
        self.graph.update(loaded)

    def export_for_viz(self, max_nodes: int = 100, min_weight: float = 0.1) -> Dict[str, List[Dict[str, object]]]:
        """
        Return a compact subgraph suitable for visualization.
        Filters edges by weight and limits node count.
        """
        strong_edges = [e for e in self.edges if abs(float(e.get("weight", 0.0))) >= min_weight]
        strong_edges.sort(key=lambda e: abs(float(e.get("weight", 0.0))), reverse=True)
        nodes: Dict[str, Dict[str, object]] = {}
        for edge in strong_edges:
            for node_id in (edge.get("source"), edge.get("target")):
                if node_id and node_id in self.nodes:
                    nodes[node_id] = self.nodes[node_id]
        # fallback: include a few nodes even if no edges pass the filter
        if not nodes:
            for node_id, node in list(self.nodes.items())[: max_nodes]:
                nodes[node_id] = node
        limited_nodes = dict(list(nodes.items())[:max_nodes])
        limited_edges = [
            e for e in strong_edges if e.get("source") in limited_nodes and e.get("target") in limited_nodes
        ][: max_nodes * 2]
        return {
            "nodes": list(limited_nodes.values()),
            "edges": limited_edges,
        }
