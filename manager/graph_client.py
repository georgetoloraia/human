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
        return data.get("neighbors", []) or data.get("matches", [])
    except Exception:
        return []
