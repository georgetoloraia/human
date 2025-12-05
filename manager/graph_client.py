from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from manager.neuron_graph import NeuronGraph

GRAPH_FILE = Path("manager/neuron_graph.json")


def load_graph(path: str | Path | None = None) -> NeuronGraph:
    target = Path(path or GRAPH_FILE)
    ng = NeuronGraph(target)
    try:
        ng.load()
    except Exception as exc:
        # Gracefully handle non-JSON/binary inputs (e.g., legacy graph.db).
        print(f"[graph_client] Failed to load graph from {target}: {exc}. Starting with empty graph.")
        ng.graph = {"version": ng.graph.get("version", "0.1"), "nodes": {}, "edges": []}
    return ng


# Lazy global graph
_GLOBAL_GRAPH = load_graph()


def _normalize_query(text: str) -> List[str]:
    parts = text.replace(",", " ").split()
    nodes = []
    for p in parts:
        if p.startswith(("plugin:", "task:", "error:", "pattern:", "env:", "concept:")):
            nodes.append(p)
        elif "." in p:
            nodes.append(f"plugin:{p}")
        else:
            nodes.append(f"error:{p}")
    return nodes


def query_graph(text: str, limit: int = 5, neighbor_limit: int = 20) -> List[Dict[str, Any]]:
    if not text:
        return []

    results: List[Dict[str, Any]] = []
    node_ids = _normalize_query(text)

    for node_id in node_ids:
        if node_id not in _GLOBAL_GRAPH.nodes:
            continue

        neighbors = _GLOBAL_GRAPH.get_neighbors(node_id, top_k=neighbor_limit)

        enriched = []
        for neighbor in neighbors:
            nid = neighbor.get("id") if isinstance(neighbor, dict) else neighbor[0]
            weight = neighbor.get("weight") if isinstance(neighbor, dict) else neighbor[1] if len(neighbor) > 1 else 0.0
            meta = _GLOBAL_GRAPH.nodes.get(nid, {})
            enriched.append(
                {
                    "id": nid,
                    "weight": weight,
                    "type": meta.get("type"),
                    "meta": meta.get("meta", {}),
                }
            )

        results.append(
            {
                "query_node": node_id,
                "node": _GLOBAL_GRAPH.nodes.get(node_id),
                "neighbors": enriched,
            }
        )

        if len(results) >= limit:
            break

    return results


# -------------------- Flask server --------------------
def create_app(graph_path: str | Path | None = None):
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Flask is required to run the graph server") from exc

    ng = load_graph(graph_path)
    app = Flask(__name__)

    @app.route("/ask", methods=["GET"])
    def ask():
        text = request.args.get("text", "").strip()
        neighbor_limit = int(request.args.get("neighbor_limit", 20))

        if not text:
            return jsonify({"error": "empty_query"})

        nodes = _normalize_query(text)
        response = []

        for node_id in nodes:
            if node_id not in ng.nodes:
                continue

            neighbors = ng.get_neighbors(node_id, top_k=neighbor_limit)
            enriched = []

            for neighbor in neighbors:
                nid = neighbor.get("id") if isinstance(neighbor, dict) else neighbor[0]
                weight = neighbor.get("weight") if isinstance(neighbor, dict) else neighbor[1] if len(neighbor) > 1 else 0.0
                meta = ng.nodes.get(nid, {})
                enriched.append(
                    {
                        "id": nid,
                        "weight": weight,
                        "type": meta.get("type"),
                        "meta": meta.get("meta", {}),
                    }
                )

            response.append(
                {
                    "query": node_id,
                    "node": ng.nodes.get(node_id),
                    "neighbors": enriched,
                }
            )

        return jsonify({"original_query": text, "results": response})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=str(GRAPH_FILE))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.graph)
    print(f"Serving graph API on http://{args.host}:{args.port} using {args.graph}")
    app.run(host=args.host, port=args.port)
