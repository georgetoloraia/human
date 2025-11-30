#!/usr/bin/env python3
"""
Bridge graph context into a local Llama (OpenAI-compatible) server.

Flow:
- Hits the local graph API `/ask` to retrieve keyword-based context from graph.db.
- Sends question + context to a local Llama server (e.g., llama.cpp `server`).
- Prints the model's answer.
"""

from __future__ import annotations

import argparse
import requests
from typing import List, Dict, Any


def fetch_context(graph_url: str, question: str, limit: int, neighbor_limit: int, neighbor_cap: int) -> str:
    resp = requests.get(
        f"{graph_url}/ask",
        params={"text": question, "limit": limit, "neighbor_limit": neighbor_limit},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    ctx_lines: List[str] = []
    # Use neighbors as rough snippets; cap to keep prompt small.
    for n in data.get("neighbors", [])[:neighbor_cap]:
        label = n.get("label", "")
        source = n.get("source", "")
        ctx_lines.append(f"{label} (from {source})")
    # If no neighbors, fall back to matches.
    if not ctx_lines:
        for m in data.get("matches", [])[:neighbor_cap]:
            label = m.get("label", "")
            source = m.get("source", "")
            ctx_lines.append(f"{label} (from {source})")
    return "\n".join(ctx_lines)


def ask_llama(llama_url: str, model: str, question: str, context: str, max_tokens: int, temperature: float) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Answer clearly using the provided context when relevant."},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]
    resp = requests.post(
        llama_url,
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Use local Llama with graph.db context.")
    ap.add_argument("question", help="Question/prompt to ask.")
    ap.add_argument("--graph-url", default="http://127.0.0.1:8000", help="Graph API base URL.")
    ap.add_argument(
        "--llama-url",
        default="http://127.0.0.1:8080/v1/chat/completions",
        help="Llama server URL (OpenAI-compatible).",
    )
    ap.add_argument("--model", default="local-llama", help="Model name expected by your Llama server.")
    ap.add_argument("--limit", type=int, default=50, help="Match budget passed to graph /ask.")
    ap.add_argument(
        "--neighbor-limit",
        type=int,
        default=200,
        help="Neighbor budget passed to graph /ask.",
    )
    ap.add_argument(
        "--neighbor-cap",
        type=int,
        default=50,
        help="Max neighbors to include in the prompt to Llama.",
    )
    ap.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate.")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    args = ap.parse_args()

    context = fetch_context(
        graph_url=args.graph_url,
        question=args.question,
        limit=args.limit,
        neighbor_limit=args.neighbor_limit,
        neighbor_cap=args.neighbor_cap,
    )
    answer = ask_llama(
        llama_url=args.llama_url,
        model=args.model,
        question=args.question,
        context=context,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()
