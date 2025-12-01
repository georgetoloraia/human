# Roadmap for the "human" Project

Goal: Evolve the **"human" project** into a self-developing, self-realizing child-mind that lives inside a computer, grows through interaction with its world (code, tasks, docs, web), and continuously improves its own skills.

This roadmap is intended for developers and for Codex-like assistants. It describes phases, constraints, and concrete tasks referencing existing files and directories in the repository.

---

## 0. CURRENT STATE (WHERE WE ARE NOW)

The "human" project already has:

### 0.1 A Mind

- `manager/mind.py` orchestrates:
  - BrainMemory (`manager/memory.py`)
  - ConceptGraph (`manager/concepts.py`, `manager/concept_graph.json`)
  - Goals (`manager/goals.py`)
  - Tasks and TaskState (`manager/tasks.py`, `manager/tasks_state.py`)
  - WebSensor and WebKnowledge (`manager/web_sensor.py`, `manager/web_knowledge.py`)
  - Reflection via local LLaMA (`manager/reflection.py`)

- The Mind loop:
  - Ages and updates stage (0..N)
  - Perceives plugins (`plugins/*.py`, skipping `__init__.py`)
  - Scores plugins with drives: curiosity, mastery, stability, task drive
  - Selects plugins and mutation candidates based on pattern scores and stage
  - Applies mutations, runs tests, evaluates, rolls back on failure
  - Tracks error streaks and consults `docs.python.org` after repeated failures
  - Logs everything to:
    - `manager/brain_memory.json`
    - `manager/tasks_state.json`
    - `manager/mind_diary.json`

### 0.2 Code World

- `plugins/` contains runnable, mutable code
- `tasks/` contains task definitions
- `tests/` contains pytest-compatible tests

### 0.3 External Perception

- Safe, whitelisted HTTP via WebSensor
- HTML → text via WebKnowledge
- Local LLaMA endpoint for per-step reflections

This is already a self-mutating, task-aware, reflective code agent.

---

## 1. PHASE 1 – STABILIZE THE CHILD AND ITS HOME

### Objective
Make the current system robust, reproducible, and easy to extend.

### 1.1 Hardening Memory and State

- Ensure `manager/memory.py` initializes all missing keys
- Truncate large arrays like `observations` and `external_knowledge`
- Ensure `manager/tasks_state.py` handles deleted/renamed tasks safely
- Add schema versioning to JSON state files

### 1.2 Deterministic Runs and Resets

- Add `scripts/reset_state.py` to reset all JSON state files
- Document how to start a new life vs resume an existing one

### 1.3 Monitoring and Basic Metrics

- Add `manager/metrics.json` for:
  - Steps taken
  - Successful/rejected mutations
  - Web consults
  - Tasks mastered

---

## 2. PHASE 2 – RICHER WORLD AND CURRICULUM

### Objective
Give the child a richer, structured world and curriculum.

### 2.1 Curriculum Schema

Extend task YAML with:
- `phase`
- `difficulty`
- `category`

### 2.2 Curriculum Manager

- Add `manager/curriculum.py`
- Track current phase, mastery, and phase transitions

### 2.3 Mind Integration

- Bias plugin selection toward current phase tasks
- Include curriculum info in `mind_diary.json`

### 2.4 Expanded Task Set

- Add more task YAMLs
- Implement test generation for all tasks

---

## 3. PHASE 3 – GRAPH KNOWLEDGE INTEGRATION

### Objective
Connect the Mind to the graph layer as its knowledge cortex.

### 3.1 Graph API Exposure

- Expand `graph_api.py` with search/query endpoints

### 3.2 Graph Client

- Add `manager/graph_client.py`

### 3.3 Graph-Assisted Reasoning

- Use graph during `_consult_web`
- Merge graph output into external knowledge

---

## 4. PHASE 4 – SELF-EXTENDING SKILLS

### Objective
Let the child invent new patterns and harder tests.

### 4.1 Pattern Generation via LLaMA

- Add `manager/pattern_generator.py`
- Track experimental patterns

### 4.2 Task Growth

- Add `manager/task_growth.py`
- Automatically generate harder tests

---

## 5. PHASE 5 – META-LEARNING AND AUTONOMY

### Objective
Evolve into a semi-autonomous, meta-learning agent.

### 5.1 Meta-Skill

- Track reflection accuracy
- Adjust exploration vs exploitation

### 5.2 Autonomous Cycles

- Detect stagnation
- Reset or branch intelligently

### 5.3 Safety Rules

- Restrict mutation targets
- Enforce domain and file permissions

---

## 6. GUIDELINES FOR CODEX / CONTRIBUTORS

- Respect the child metaphor
- Prefer modular, composable systems
- Avoid breaking state format
- Test after each phase

---

## FINAL VISION

"human" becomes a living playground where a small artificial child-mind grows inside the computer, starting from almost nothing and slowly learning programming, reasoning, and self-improvement through real interaction with its world, tasks, knowledge graph, and human guidance.

