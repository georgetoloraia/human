# Project Overview
This repo has two layers:
- **Graph tools**: build/query a neuron graph from text (nn_builder, query, graph_api).
- **Self-growing mind**: an agent that mutates `plugins/*.py` to satisfy tasks, with stages, drives, tasks, web consults, and a diary.

# Setup
Requires Python 3. Install deps:
```bash
pip install -r requirements.txt
```
If using Postgres for the graph layer, also install `psycopg` or `psycopg2`.
Ensure a local LLaMA/OpenAI-compatible endpoint is running if you want reflections and web consults logged (see below).

# Graph layer (optional)
Build/lint:
```bash
python3 -m py_compile nn_builder.py query.py graph_api.py
```
Ingest text to SQLite:
```bash
python3 nn_builder.py --text-files path/to/texts/* --sqlite-out graph.db --max-neurons 300000 --max-tokens-per-file 50000 --bridge-limit 50000
```
Start HTTP API:
```bash
python3 graph_api.py --db graph.db --host 127.0.0.1 --port 8000
```
Query:
```bash
python3 query.py --db graph.db socket
curl "http://127.0.0.1:8000/ask?text=How%20do%20I%20work%20with%20sockets%20in%20Python?&limit=50&neighbor_limit=200"
```

# Self-growing mind
What it does:
- Observes `plugins/*.py` (skips __init__.py).
- Targets plugins based on drives: curiosity, mastery, stability, and task needs; stages alter breadth/depth.
- Mutates code using pattern-based proposals; tests with pytest; rolls back on failures.
- Tracks tasks from `tasks/*.yml` and per-task streaks in `manager/tasks_state.json`.
- Tracks error streaks; on repeated failures (stage ≥2) consults whitelisted Python docs and stores text.
- Writes a diary with selections, actions, tasks, reflections (via LLaMA).

Minimal world provided:
- `plugins/sample_plugin.py`
- `tasks/list_sum.yml`

Run the loop:
```bash
python3 -m manager.life_loop
```

Files/state:
- `manager/brain_memory.json`: observations, pattern stats, external knowledge, error streaks.
- `manager/concept_graph.json`: plugin/function stats.
- `manager/curriculum_state.json`: curriculum phases (if used).
- `manager/tasks_state.json`: per-task streaks and status.
- `manager/mind_diary.json`: per-step thoughts, actions, tasks, reflections.
- `manager/metrics.json`: counts of steps, accepted/rejected mutations, web consults, tasks mastered.

Resetting state (fresh life):
```bash
python3 scripts/reset_state.py
```
This clears state JSONs and recreates sample plugin/task.

LLaMA reflections:
- Configure env vars if not default:
  - `MIND_LLM_BASE_URL` (default `http://127.0.0.1:11434/v1/chat/completions`)
  - `MIND_LLM_MODEL` (default `llama3.2:1b`)
- Ensure the server is running; otherwise reflections will be empty.

Web consults:
- Whitelisted domains: docs.python.org, realpython.com, developer.mozilla.org.
- Triggered after repeated failures on the same error type (stage ≥2). Stored under `external_knowledge` and logged in actions.

Tasks format (`tasks/*.yml`):
```yaml
name: list_sum
phase: 1
target_plugin: sample_plugin.py
target_function: process
description: The function must return the sum of a list of integers.
requirements:
  - process([1,2,3]) == 6
  - process([]) == 0
  - process([-1,1]) == 0
```

License
See LICENSE (all rights reserved by George Toloraia).
