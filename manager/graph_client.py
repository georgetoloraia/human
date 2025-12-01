from __future__ import annotations

import os
from typing import List, Dict, Any

import requests


GRAPH_API_URL = os.getenv("GRAPH_API_URL", "http://127.0.0.1:8000")


def query_graph(query: str, limit: int = 5, neighbor_limit: int = 50) -> List[Dict[str, Any]]:
    """
    Query the local graph API /ask endpoint for a natural language phrase.
    Returns matches and neighbors as a list of dicts.
    """
    url = f"{GRAPH_API_URL}/ask"
    try:
        resp = requests.get(
            url,
            params={"text": query, "limit": limit, "neighbor_limit": neighbor_limit},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        neighbors = data.get("neighbors", []) or data.get("matches", [])
        # normalize fields
        norm = []
        for n in neighbors:
            norm.append(
                {
                    "id": n.get("id"),
                    "label": n.get("label"),
                    "source": n.get("source"),
                }
            )
        return norm
    except Exception:
        return []


def fetch_neighbors(node_ids: List[int], limit: int = 20) -> List[Dict[str, Any]]:
    """
    Placeholder neighbor fetch; if graph_api exposes such an endpoint, wire it here.
    Currently reuses /query by label lookup if ids are known.
    """
    results: List[Dict[str, Any]] = []
    if not node_ids:
        return results
    # With current graph_api, we can't fetch by id; this is a stub for future expansion.
    return results
