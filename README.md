# Brain Graph Builder

Scripts to ingest text into a simple neuron/edge graph, persist it (SQLite/Postgres), query it from the CLI, and serve a tiny HTTP API.

## What this project can do
- Turn text files into "neurons" (tokens) with edges linking tokens in sequence and bridging across files.
- Persist the graph to SQLite (`--sqlite-out`) or Postgres (`--pg-url`) to avoid holding everything in RAM.
- Query by token via CLI (`query.py`) or HTTP (`graph_api.py`), including a basic natural-language-ish `/ask` endpoint.

## Setup
Requires Python 3. Install deps if you plan to use Postgres:
```bash
pip install psycopg psycopg2  # one of them if not already installed
```

## Build / lint
```bash
python3 -m py_compile nn_builder.py query.py graph_api.py
```

## Ingest text into a graph
Interactive (numeric network mode):
```bash
python3 nn_builder.py --interactive --demo-epochs 3
```

Text mode to SQLite (recommended for large corpora):
```bash
python3 nn_builder.py \
  --text-files path/to/texts/* \
  --sqlite-out graph.db \
  --max-neurons 300000 \
  --max-tokens-per-file 50000 \
  --bridge-limit 50000
```

Text mode to Postgres:
```bash
python3 nn_builder.py \
  --text-files path/to/texts/* \
  --pg-url "postgres://user:pass@host:5432/db" \
  --pg-schema public \
  --max-neurons 300000 \
  --max-tokens-per-file 50000
```
Notes: Postgres must be reachable; tables `neurons` and `edges` (with indexes) are created automatically. Each run starts neuron IDs at 0; reuse the same DB only if you want to overwrite IDs.

Key flags:
- `--max-neurons`: global cap; `-1` disables.
- `--max-tokens-per-file`: per-file cap; `-1` disables.
- `--bridge-limit`: caps bridge edges between prior and new neurons; `-1` disables.
- `--text-files` accepts files and directories (walked recursively).

## Query via CLI (SQLite)
```bash
python3 query.py --db graph.db socket
python3 query.py --db graph.db --match-like sock --limit 100 --neighbor-limit 300
```
Shows matched neurons and outgoing neighbors with source counts.

## HTTP API
Start the server:
```bash
python3 graph_api.py --db graph.db --host 127.0.0.1 --port 8000
```
Endpoints:
- `GET /health` -> `{"status":"ok"}`
- `GET /query?term=foo&like=1&limit=50&neighbor_limit=200`
- `GET /ask?text=How+do+I+work+with+sockets?&limit=50&neighbor_limit=200`

`/ask` tokenizes the text, filters stopwords/short tokens, dedupes, and uses the `limit` as a total budget across tokens.

## Self-growing loop (experimental)
Files under `manager/` implement a small self-evolution loop with persistence in `manager/brain_memory.json` and a simple concept/goal layer stored in `manager/concept_graph.json`.

Run the loop:
```bash
python3 -m manager.life_loop
```
- Observes `plugins/*.py`, records observations in memory.
- Grows functions by inserting safe lines.
- Builds a concept graph of plugins/functions and tracks growth/tests to prioritize where to grow next.
- Uses `manager/safety.py` to reject syntactically invalid or oversized mutations.
- Uses `manager/tester.py` and `manager/evaluator.py` stubs to accept/reject mutations (customize these for real checks).
- Rolls back on failure; increments skill/age on success.

Add your own plugin files under `plugins/` to see mutations in action. Adjust the sleep interval or logic in `manager/life_loop.py` as needed.

## Use a local Llama as the "voice"
Run your Llama server (OpenAI-compatible, e.g., llama.cpp `server`):
```bash
./server -m /path/to/model.gguf -c 4096 --port 8080  # adjust to your setup
```
Then ask questions with graph context:
```bash
python3 answer.py "How do I work with sockets in Python?" \
  --graph-url http://127.0.0.1:8000 \
  --llama-url http://127.0.0.1:8080/v1/chat/completions \
  --model local-llama \
  --neighbor-cap 50
```
`answer.py` hits `/ask` on the graph API to gather context, then prompts your Llama server to produce a human-readable answer.

## Working with graph.db elsewhere
- Use `sqlite3 graph.db "SELECT COUNT(*) FROM neurons;"` or copy the DB to another project.
- Example Python snippet:
```python
import sqlite3
conn = sqlite3.connect("graph.db")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM neurons")
print(cur.fetchone()[0])
conn.close()
```

## Caveats
- IDs restart at 0 each run; reuse the same sink only if you intend to overwrite matching IDs.
- Bridge edges can grow fast; adjust `--bridge-limit` and neuron caps for large corpora.
- `/ask` is keyword-based; for semantic search you'd add embeddings/vector search on top.
