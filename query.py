"""
Lightweight query helper for neuron graphs stored in SQLite.

Given a search term and a SQLite database (graph.db), it fetches matching
neurons and a small neighborhood of connected neurons to provide context.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query neurons and neighbor snippets from a neuron graph SQLite DB."
    )
    parser.add_argument(
        "--db",
        default="graph.db",
        help="Path to SQLite DB produced by nn_builder.py (--sqlite-out). Default: graph.db",
    )
    parser.add_argument(
        "term",
        help="Search term to match against neuron labels (case-sensitive exact match).",
    )
    parser.add_argument(
        "--match-like",
        action="store_true",
        help="Use SQL LIKE (%%term%%) instead of exact match on label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max neurons to return for the term.",
    )
    parser.add_argument(
        "--neighbor-limit",
        type=int,
        default=200,
        help="Max neighbor neurons to pull for the matched set.",
    )
    return parser.parse_args()


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    try:
        return sqlite3.connect(str(db_path))
    except Exception as exc:
        raise SystemExit(f"Failed to open DB: {exc}")


def fetch_ids(cur: sqlite3.Cursor, term: str, like: bool, limit: int) -> list[int]:
    if like:
        cur.execute(
            "SELECT id FROM neurons WHERE label LIKE ? LIMIT ?",
            (f"%{term}%", limit),
        )
    else:
        cur.execute("SELECT id FROM neurons WHERE label = ? LIMIT ?", (term, limit))
    return [row[0] for row in cur.fetchall()]


def fetch_neighbors(
    cur: sqlite3.Cursor, ids: Iterable[int], limit: int
) -> list[tuple[int, str, str]]:
    ids = list(ids)
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


def main() -> None:
    args = parse_args()
    conn = connect(Path(args.db))
    cur = conn.cursor()

    ids = fetch_ids(cur, args.term, args.match_like, args.limit)
    if not ids:
        print("No matching neurons.")
        return
    print(f"Matched {len(ids)} neurons for term '{args.term}'.")

    cur.execute(
        f"SELECT id, label, source FROM neurons WHERE id IN ({','.join('?' for _ in ids)})",
        ids,
    )
    matches = cur.fetchall()

    neighbors = fetch_neighbors(cur, ids, args.neighbor_limit)

    def summarize(records: list[tuple[int, str, str]], title: str, max_rows: int = 20) -> None:
        print(f"\n{title} (showing up to {max_rows}):")
        for rid, label, source in records[:max_rows]:
            print(f"- id={rid} label={label} source={source}")
        if len(records) > max_rows:
            print(f"... {len(records) - max_rows} more")

    summarize(matches, "Matches", max_rows=min(20, args.limit))
    summarize(neighbors, "Neighbor neurons via outgoing edges", max_rows=min(20, args.neighbor_limit))

    by_source = defaultdict(int)
    for _, _, src in neighbors:
        by_source[src] += 1
    if by_source:
        print("\nNeighbor counts by source (top 10):")
        for src, count in sorted(by_source.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"- {src}: {count}")

    conn.close()


if __name__ == "__main__":
    main()
