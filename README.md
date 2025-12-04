# Project Overview
Two layers live here:
- **Graph tools**: build/query a neuron graph from text (`nn_builder.py`, `query.py`, `graph_api.py`).
- **Self-growing mind**: an agent that mutates `plugins/*.py` to satisfy tasks, with stages, drives, tasks, web consults, reflections, and a diary.

# Setup
Python 3 required. Install deps:
```bash
pip install -r requirements.txt
```
If using Postgres for graph ingestion, also install `psycopg`/`psycopg2`. Run a local LLaMA-compatible endpoint for reflections (config below).

# Graph layer (optional)
- Build/lint: `python3 -m py_compile nn_builder.py query.py graph_api.py`
- Ingest to SQLite: `python3 nn_builder.py --text-files path/* --sqlite-out graph.db --max-neurons 300000 --max-tokens-per-file 50000 --bridge-limit 50000`
- Serve API: `python3 graph_api.py --db graph.db --host 127.0.0.1 --port 8000`
- Query: `python3 query.py --db graph.db socket` or `curl "http://127.0.0.1:8000/ask?text=..."`.

# Self-growing mind
Run the loop:
```bash
python3 -m manager.life_loop
```
Behaviors:
- Perceives `plugins/*.py` (skips __init__.py), scores plugins with drives (curiosity, mastery, stability, task-drive) and stages.
- Mutates code via patterns; tests with pytest; rolls back on failure.
- Tracks tasks from `tasks/*.yml` (phase/difficulty/category), per-task streaks in `manager/tasks_state.json`; advances phases when enough tasks are mastered.
- Tracks error streaks; on repeated failures (stage ≥2) consults whitelisted docs and graph API; stores text in external knowledge.
- Logs per-step diary with selections, actions, tasks, curriculum, reflections (via LLaMA), lifecycle events, meta-skill.
- Metrics recorded in `manager/metrics.json`.

State files:
- `manager/brain_memory.json` (observations, patterns, external knowledge, error/meta-skill).
- `manager/concept_graph.json` (plugin/function stats).
- `manager/curriculum_state.json` (phase/mastery).
- `manager/tasks_state.json` (task streaks/status).
- `manager/mind_diary.json` (per-step log).
- `manager/metrics.json` (steps, accepted/rejected, consults, tasks mastered).

Reset (fresh life):
```bash
python3 scripts/reset_state.py
```
Clears state JSONs and recreates sample plugin/task.

LLaMA/reflection config:
- `MIND_LLM_BASE_URL` (default `http://127.0.0.1:11434/v1/chat/completions`)
- `MIND_LLM_MODEL` (default `llama3.2:1b`)

Web/graph consults:
- Domains: docs.python.org, realpython.com, developer.mozilla.org.
- Graph API: `GRAPH_API_URL` (default `http://127.0.0.1:8000`).

Task format (`tasks/*.yml`):
```yaml
name: list_sum
phase: 1
difficulty: 1
category: math
target_plugin: sample_plugin.py
target_function: process
description: ...
requirements:
  - process([1,2,3]) == 6
  - process([]) == 0
  - process([-1,1]) == 0
```

License: see LICENSE (all rights reserved by George Toloraia).

# Architecture additions
- **NeuronGraph** (`manager/neuron_graph.py`): embedding-backed nodes for plugins/tasks/errors/reflections with weighted edges and Hebbian updates; persisted to `manager/neuron_graph.json`.
- **Embeddings + Value Function** (`manager/embeddings.py`, `manager/value_function.py`): pluggable embeddings with offline fallback and reward-informed scoring (`USE_VALUE_FUNCTION=1`).
- **Multi-agent loop** (`manager/agents/`): planner → coder → critic → memory agents orchestrated inside the life loop when `HUMAN_MULTI_AGENT=1`.
- **Online learning + reward** (`manager/reward.py`, metrics extensions): moving averages per plugin/strategy plus neuron-graph edge updates driven by rewards.
- **Perception & practice envs** (`manager/envs/`): structured error/percept logging and lightweight text/symbol environments (`HUMAN_ENABLE_ENVS=1`, `HUMAN_ENV_RATIO`).
- **Reward shaping v2**: structured outcome taxonomy (domain/tests/env progress) with clipped rewards and breakdowns stored in metrics history.
- **Trace/debug + async scaffolding**: set `HUMAN_TRACE=1` to emit per-step traces (candidate scores, actions, rewards) under `manager/traces/`; set `HUMAN_ASYNC=1` to use the async orchestrator wrapper for perception/env steps.
- **Env curriculum scheduling**: `HUMAN_ENV_RATIO` controls how often env steps run alongside code steps; domain choices are recorded in metrics for analysis and visualization (see `export_for_viz` in the neuron graph).

# Reset and rerun
- Reset state (backs up JSONs to `backups/<timestamp>/` and removes them):  
  `python3 scripts/reset_state.py`
- Fresh run with tracing and value function, code-only:  
  ```HUMAN_MULTI_AGENT=1 HUMAN_TRACE=1 USE_VALUE_FUNCTION=1 HUMAN_ASYNC=0 HUMAN_ENV_RATIO=0.0 python3 -m manager.life_loop```
- Analyze the latest run:  
  ```python3 scripts/analyze_brain_run.py```
