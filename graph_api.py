"""
Minimal HTTP API for querying a neuron graph stored in SQLite.

Endpoints:
- GET /health            -> {"status": "ok"}
- GET /query?term=foo    -> returns matches and neighbor neurons.
  Optional params:
    like=1               -> use SQL LIKE %term% instead of exact match
    limit=50             -> max matches
    neighbor_limit=200   -> max neighbors via outgoing edges

Run:
    python3 graph_api.py --db graph.db --host 127.0.0.1 --port 8000

This uses only the standard library (http.server + sqlite3).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a simple HTTP API over graph.db")
    parser.add_argument("--db", default="graph.db", help="Path to SQLite DB (default: graph.db)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--default-limit", type=int, default=50, help="Default max matches per query (default: 50)"
    )
    parser.add_argument(
        "--default-neighbors",
        type=int,
        default=200,
        help="Default max neighbors per query (default: 200)",
    )
    return parser.parse_args()


def connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def fetch_ids(cur: sqlite3.Cursor, term: str, like: bool, limit: int) -> list[int]:
    if like:
        cur.execute(
            "SELECT id FROM neurons WHERE label LIKE ? LIMIT ?",
            (f"%{term}%", limit),
        )
    else:
        cur.execute("SELECT id FROM neurons WHERE label = ? LIMIT ?", (term, limit))
    return [row[0] for row in cur.fetchall()]


def fetch_records(cur: sqlite3.Cursor, ids: list[int]) -> list[tuple[int, str, str]]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    cur.execute(
        f"SELECT id, label, source FROM neurons WHERE id IN ({placeholders})",
        ids,
    )
    return cur.fetchall()


def fetch_neighbors(cur: sqlite3.Cursor, ids: list[int], limit: int) -> list[tuple[int, str, str]]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    cur.execute(
        f"""
        SELECT n2.id, n2.label, n2.source
        FROM edges e
        JOIN neurons n2 ON n2.id = e.dst
        WHERE e.src IN ({placeholders})
        LIMIT ?
        """,
        (*ids, limit),
    )
    return cur.fetchall()


STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "be",
    "this",
    "that",
    "it",
    "its",
    "as",
    "at",
    "from",
    "of",
    "do",
    "did",
    "done",
    "does",
    "i",
    "you",
    "your",
    "we",
    "they",
    "them",
    "their",
    "our",
    "ours",
    "how",
    "what",
    "why",
    "when",
    "where",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "have",
    "has",
    "had",
    "been",
    "being",
    "if",
    "then",
    "else",
    "so",
    "not",
    "no",
    "yes",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    filtered = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) < 3:
            continue
        filtered.append(t)
    return filtered


class GraphRequestHandler(BaseHTTPRequestHandler):
    db_path: str
    default_limit: int
    default_neighbors: int

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server signature
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/query":
            params = parse_qs(parsed.query)
            term = params.get("term", [None])[0]
            if not term:
                self._send_json({"error": "missing term"}, status=400)
                return
            like = params.get("like", ["0"])[0] in {"1", "true", "yes"}
            try:
                limit = int(params.get("limit", [self.default_limit])[0])
                neighbor_limit = int(params.get("neighbor_limit", [self.default_neighbors])[0])
            except ValueError:
                self._send_json({"error": "limit and neighbor_limit must be integers"}, status=400)
                return

            try:
                conn = connect(self.db_path)
                cur = conn.cursor()
                ids = fetch_ids(cur, term, like=like, limit=limit)
                matches = fetch_records(cur, ids)
                neighbors = fetch_neighbors(cur, ids, neighbor_limit)
                conn.close()
            except Exception as exc:
                self._send_json({"error": f"query failed: {exc}"}, status=500)
                return

            src_counts = {}
            for _, _, src in neighbors:
                src_counts[src] = src_counts.get(src, 0) + 1

            def to_dict(rec: tuple[int, str, str]) -> dict:
                rid, label, source = rec
                return {"id": rid, "label": label, "source": source}

            payload = {
                "term": term,
                "like": like,
                "matches": [to_dict(r) for r in matches],
                "neighbors": [to_dict(r) for r in neighbors],
                "stats": {
                    "match_count": len(matches),
                    "neighbor_count": len(neighbors),
                    "sources": src_counts,
                },
            }
            self._send_json(payload)
            return
        if parsed.path == "/ask":
            params = parse_qs(parsed.query)
            text = params.get("text", [None])[0]
            if not text:
                self._send_json({"error": "missing text"}, status=400)
                return
            try:
                limit = int(params.get("limit", [self.default_limit])[0])
                neighbor_limit = int(params.get("neighbor_limit", [self.default_neighbors])[0])
            except ValueError:
                self._send_json({"error": "limit and neighbor_limit must be integers"}, status=400)
                return

            tokens = tokenize(text)
            if not tokens:
                self._send_json({"error": "no usable tokens in text"}, status=400)
                return
            # Deduplicate while preserving order.
            seen_tokens = set()
            deduped_tokens: list[str] = []
            for tok in tokens:
                if tok not in seen_tokens:
                    seen_tokens.add(tok)
                    deduped_tokens.append(tok)

            try:
                conn = connect(self.db_path)
                cur = conn.cursor()
                seen_ids = set()
                collected_ids: list[int] = []
                remaining = max(0, limit)
                for tok in deduped_tokens:
                    if remaining <= 0:
                        break
                    tok_ids = fetch_ids(cur, tok, like=False, limit=remaining)
                    for nid in tok_ids:
                        if nid not in seen_ids:
                            seen_ids.add(nid)
                            collected_ids.append(nid)
                            remaining -= 1
                            if remaining <= 0:
                                break
                matches = fetch_records(cur, collected_ids)
                neighbors = fetch_neighbors(cur, collected_ids, neighbor_limit)
                conn.close()
            except Exception as exc:
                self._send_json({"error": f"ask failed: {exc}"}, status=500)
                return

            src_counts = {}
            for _, _, src in neighbors:
                src_counts[src] = src_counts.get(src, 0) + 1

            def to_dict(rec: tuple[int, str, str]) -> dict:
                rid, label, source = rec
                return {"id": rid, "label": label, "source": source}

            payload = {
                "text": text,
                "tokens": deduped_tokens,
                "matches": [to_dict(r) for r in matches],
                "neighbors": [to_dict(r) for r in neighbors],
                "stats": {
                    "match_count": len(matches),
                    "neighbor_count": len(neighbors),
                    "sources": src_counts,
                },
            }
            self._send_json(payload)
            return

        self._send_json({"error": "not found"}, status=404)


def run_server(db_path: str, host: str, port: int, default_limit: int, default_neighbors: int) -> None:
    handler = GraphRequestHandler
    handler.db_path = db_path
    handler.default_limit = default_limit
    handler.default_neighbors = default_neighbors

    server = HTTPServer((host, port), handler)
    print(f"Serving graph API on http://{host}:{port} using DB {db_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    run_server(
        db_path=args.db,
        host=args.host,
        port=args.port,
        default_limit=args.default_limit,
        default_neighbors=args.default_neighbors,
    )


if __name__ == "__main__":
    main()
