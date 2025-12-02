from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

from manager.memory import BrainMemory
from manager.concepts import ConceptGraph
from manager.goals import Goals
from manager.generator import propose_mutations
from manager.safety import is_safe
from manager.tester import run_tests
from manager.evaluator import evaluate
from manager.perception import observe_codebase
from manager.task_tests import regenerate_task_tests
from manager.curriculum import Curriculum
from manager.tasks import load_tasks
from manager.tasks_state import TaskStateManager
from manager.web_sensor import WebSensor
from manager.web_knowledge import extract_plain_text
from manager.reflection import generate_reflection
from manager.metrics import Metrics
from manager.meta_policy import MetaPolicy
from manager.graph_client import query_graph
from manager.guidance import latest_guidance
from manager.doc_mastery import compute_and_save_doc_mastery, load_doc_mastery_state
from manager.doc_index import load_concepts
from manager.concept_miner import choose_next_concept
from manager.task_grower import ensure_tasks_for_concept

PLUGINS_DIR = Path("plugins")
DIARY_FILE = Path("manager/mind_diary.json")


class Mind:
    """
    A unified 'child mind' that:
      - ages and maintains an internal stage (development level)
      - perceives the world (plugins + structure)
      - evaluates plugins using curiosity / mastery / stability drives
      - chooses where to act
      - mutates code using learned patterns
      - learns from feedback (tests, errors)
      - writes a 'thought diary' about what it did and why
    """

    def __init__(self) -> None:
        self.brain = BrainMemory()
        self.graph = ConceptGraph()
        self.curriculum = Curriculum()
        self.tasks = load_tasks()
        self.task_state = TaskStateManager(self.tasks)
        self.goals = Goals(self.graph, self.curriculum)
        self.web_sensor = WebSensor()
        self.metrics = Metrics()
        self.meta_policy = MetaPolicy(self.brain)
        self.stage: int = 0
        self.last_error_type: str | None = None
        self._last_selection_info: List[Dict[str, Any]] = []
        self._current_step_actions: List[Dict[str, Any]] = []
        self._stagnation_steps: int = 0
        self._last_mastered: int = 0
        self._last_step_stats: Dict[str, Any] = {}
        self._lifecycle_event: Dict[str, Any] | None = None
        self.doc_concepts = load_concepts()
        self.doc_curriculum_enabled = os.getenv("HUMAN_DOC_CURRICULUM", "1") != "0"
        self.doc_mastery = load_doc_mastery_state()
        self._doc_curriculum_events: List[Dict[str, Any]] = []

    def step(self) -> None:
        self._last_selection_info = []
        self._current_step_actions = []
        self._doc_curriculum_events = []
        self._age_and_stage()
        self._maybe_expand_doc_curriculum()
        current_age = self.brain.state.get("age", 0)
        current_skill = self.brain.get_skill_level()
        tasks = regenerate_task_tests(self.curriculum)
        self.graph.register_tasks(tasks)
        active_task_names = [t["name"] for t in tasks if "name" in t]
        self._perceive_world()
        self._act_and_learn(active_task_names)
        self._update_metrics()
        self.curriculum.sync_with_task_state(self.task_state.state)
        if self.curriculum.should_advance_phase():
            self.curriculum.advance_phase()
        self._update_meta_skill()
        meta_status = self.meta_policy.tick(current_age, current_skill)
        self._detect_stagnation()
        self._update_doc_mastery()
        self._log_step_thought()

    def _age_and_stage(self) -> None:
        self.brain.grow()
        age = self.brain.state.get("age", 0)
        skill = self.brain.get_skill_level()
        if skill < 5 and age < 20:
            self.stage = 0
        elif skill < 20:
            self.stage = 1
        elif skill < 50:
            self.stage = 2
        else:
            self.stage = 3

    def _doc_curriculum_ready(self) -> bool:
        if not self.doc_curriculum_enabled:
            return False
        state = getattr(self.task_state, "state", {})
        if not state:
            return False
        total_passes = sum(int(info.get("passes", 0) or 0) for name, info in state.items() if not name.startswith("_"))
        if total_passes < 5:
            return False
        core_tasks = ["easy_pass", "list_sum", "add_two_numbers"]
        mastered_core = 0
        for task in core_tasks:
            info = state.get(task, {})
            if info.get("last_status") == "passing" and info.get("streak", 0) > 0:
                mastered_core += 1
        return mastered_core >= 2

    def _maybe_expand_doc_curriculum(self) -> None:
        if not self._doc_curriculum_ready():
            return
        concept = choose_next_concept(self.task_state, self.doc_concepts, self.doc_mastery)
        if not concept:
            return
        created = ensure_tasks_for_concept(concept)
        if created:
            self.tasks = load_tasks()
            self.task_state = TaskStateManager(self.tasks)
            event = {
                "action": "added_concept",
                "concept": concept.get("id") or concept.get("name"),
                "task_files": created,
                "doc_snippet": concept.get("doc_snippet"),
            }
            self._doc_curriculum_events.append(event)
    def _perceive_world(self) -> None:
        for p in PLUGINS_DIR.glob("*.py"):
            if p.name == "__init__.py":
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except Exception:
                continue
            obs = {
                "file": p.name,
                "lines": len(src.splitlines()),
                "has_functions": "def " in src,
                "has_return": "return" in src,
            }
            self.brain.observe(obs)
            self.graph.observe_plugin(p)

    def _score_plugin(self, plugin_name: str) -> Dict[str, float]:
        p = self.graph.graph.get("plugins", {}).get(plugin_name, {})
        tests_passed = float(p.get("tests_passed", 0))
        tests_failed = float(p.get("tests_failed", 0))
        growth_count = float(p.get("growth_count", 0))
        curiosity = 1.0 / (1.0 + growth_count)
        mastery = tests_passed - tests_failed
        stability_penalty = -abs(tests_failed)
        tstats = self.task_state.plugin_task_stats(plugin_name)
        task_count = tstats["task_count"]
        avg_streak = tstats["avg_streak"]
        failing_count = tstats["failing_count"]
        task_need = failing_count
        task_mastery_component = -avg_streak
        task_drive = task_need + task_mastery_component
        current_phase = self.curriculum.state.get("current_phase", 1)
        phase_boost = 0.0
        future_penalty = 0.0
        for tname, info in self.task_state.state.items():
            if tname.startswith("_"):
                continue
            if info.get("phase") == current_phase and info.get("last_status") == "failing":
                if info.get("plugin") == plugin_name:
                    phase_boost += 1.0
            if info.get("phase", 1) > current_phase and info.get("plugin") == plugin_name:
                future_penalty += 1.0
        if self.stage == 0:
            w_c, w_m, w_s, w_t = 2.0, 0.5, 0.5, 0.3
        elif self.stage == 1:
            w_c, w_m, w_s, w_t = 1.5, 1.2, 1.0, 1.0
        elif self.stage == 2:
            w_c, w_m, w_s, w_t = 0.8, 2.0, 1.0, 1.8
        else:
            w_c, w_m, w_s, w_t = 0.5, 2.5, 1.5, 2.5
        total = (
            w_c * curiosity
            + w_m * mastery
            + w_s * stability_penalty
            + w_t * (task_drive + phase_boost - future_penalty)
        )
        return {
            "curiosity": curiosity,
            "mastery": mastery,
            "stability_penalty": stability_penalty,
            "task_count": task_count,
            "task_avg_streak": avg_streak,
            "task_failing_count": failing_count,
            "task_drive": task_drive + phase_boost - future_penalty,
            "total": total,
        }

    def _select_targets(self) -> List[str]:
        base_targets = self.goals.choose_plugins_to_grow(max_plugins=4)
        if not base_targets:
            base_targets = [p.name for p in PLUGINS_DIR.glob("*.py") if p.name != "__init__.py"]
        scored: List[Tuple[float, str, Dict[str, float]]] = []
        current_phase = self.curriculum.current_phase()
        for name in base_targets:
            scores = self._score_plugin(name)
            scored.append((scores["total"], name, scores))
        scored.sort(reverse=True, key=lambda x: x[0])
        policy = self.brain.get_learning_policy()
        policy_max_plugins = int(policy.get("max_plugins_per_step", 2))
        if self.stage == 0:
            max_plugins = min(3, policy_max_plugins)
        elif self.stage == 1:
            max_plugins = min(2, policy_max_plugins)
        else:
            max_plugins = min(1, policy_max_plugins)
        selected = scored[: max(1, max_plugins)]
        self._last_selection_info = [
            {
                "plugin": name,
                "total_score": total,
                "curiosity": scores["curiosity"],
                "mastery": scores["mastery"],
                "stability_penalty": scores["stability_penalty"],
            }
            for (total, name, scores) in selected
        ]
        return [name for (total, name, scores) in selected]

    def _act_and_learn(self, active_task_names: List[str]) -> None:
        pattern_scores = self.brain.pattern_scores()
        targets = self._select_targets()
        policy = self.brain.get_learning_policy()
        depth_override = int(policy.get("exploration_depth", 3))
        if self.stage == 0:
            max_candidates_per_plugin = min(5, depth_override)
        elif self.stage == 1:
            max_candidates_per_plugin = min(3, depth_override)
        elif self.stage == 2:
            max_candidates_per_plugin = min(2, depth_override)
        else:
            max_candidates_per_plugin = min(1, depth_override)
        meta = self.brain.state.get("meta_skill", 0.0)
        if meta > 0.6:
            max_candidates_per_plugin = max(1, max_candidates_per_plugin - 1)
        elif meta < 0.3:
            max_candidates_per_plugin = min(5, max_candidates_per_plugin + 1)

        for plugin_name in targets:
            path = PLUGINS_DIR / plugin_name
            if not path.exists():
                continue
            src = path.read_text(encoding="utf-8")
            error_scores = self.brain.pattern_error_scores(self.last_error_type) if self.last_error_type else {}
            last_consult = self.brain.get_last_consult()
            if last_consult.get("error_type") and last_consult.get("error_type") == self.last_error_type:
                # boost patterns that historically work for this error type
                boosted = {}
                base = self.brain.pattern_error_scores(self.last_error_type)
                for name, score in base.items():
                    boosted[name] = score * 1.5
                error_scores.update(boosted)
            candidates = propose_mutations(src, pattern_scores, error_scores)
            if not candidates:
                continue
            candidates = candidates[:max_candidates_per_plugin]
            for new_code, pattern_name in candidates:
                accepted = self._try_candidate(path, plugin_name, new_code, pattern_name, active_task_names)
                if accepted:
                    break

    def _try_candidate(
        self,
        path: Path,
        plugin_name: str,
        new_code: str,
        pattern_name: str,
        active_task_names: List[str],
    ) -> bool:
        if not is_safe(new_code):
            self.brain.record_pattern_result(pattern_name, False)
            self.graph.record_test_result(plugin_name, False)
            self.task_state.record_plugin_result(plugin_name, success=False, error_type="Unsafe")
            self.brain.record_error_event(plugin_name, "Unsafe", success=False)
            self._current_step_actions.append(
                {"plugin": plugin_name, "pattern": pattern_name, "result": "unsafe", "tests_ok": False, "error_type": "Unsafe"}
            )
            return False

        orig = path.read_text(encoding="utf-8")
        path.write_text(new_code, encoding="utf-8")

        tests_ok, err_type, _failing_tasks, task_outcomes = run_tests(plugin_name=plugin_name)
        eval_ok = tests_ok and evaluate(orig, new_code)

        mutation_id = f"{plugin_name}:{pattern_name}"

        self.graph.record_test_result(plugin_name, tests_ok)
        task_results = {}
        for tname in active_task_names:
            passed = task_outcomes.get(tname, (True, "OK"))[0]
            task_results[tname] = passed
            self.graph.record_task_result(tname, passed)
        for tname, (passed, _) in task_outcomes.items():
            if tname not in task_results:
                task_results[tname] = passed
                self.graph.record_task_result(tname, passed)
        self.curriculum.update_results(task_results)
        if task_outcomes:
            self.task_state.record_task_results(task_outcomes)
        self.brain.record_error_event(plugin_name, err_type, success=eval_ok)
        streak = self.brain.get_error_streak(plugin_name, err_type)
        web_consult_info = None
        if (not eval_ok) and self.stage >= 2 and err_type != "OK":
            if streak >= 3:
                web_consult_info = self._consult_web(plugin_name, err_type)

        if eval_ok:
            self.brain.record_attempt(mutation_id, True)
            self.brain.record_pattern_result(pattern_name, True)
            self.brain.record_pattern_error_result(pattern_name, err_type, True)
            self.graph.record_growth(plugin_name)
            if not tests_ok:
                self.last_error_type = err_type
            self._current_step_actions.append(
                {
                    "plugin": plugin_name,
                    "pattern": pattern_name,
                    "result": "accepted",
                    "tests_ok": tests_ok,
                    "error_type": err_type,
                    **({"web_consult": web_consult_info} if web_consult_info else {}),
                }
            )
            return True

        path.write_text(orig, encoding="utf-8")
        self.brain.record_attempt(mutation_id, False)
        self.brain.record_pattern_result(pattern_name, False)
        self.brain.record_pattern_error_result(pattern_name, err_type, False)
        self.last_error_type = err_type
        self._current_step_actions.append(
            {
                "plugin": plugin_name,
                "pattern": pattern_name,
                "result": "rejected",
                "tests_ok": tests_ok,
                "error_type": err_type,
                **({"web_consult": web_consult_info} if web_consult_info else {}),
            }
        )
        return False

    def _log_step_thought(self) -> None:
        age = self.brain.state.get("age", 0)
        skill = self.brain.get_skill_level()
        guidance_msgs = latest_guidance()
        entry: Dict[str, Any] = {
            "age": age,
            "stage": self.stage,
            "skill": skill,
            "current_phase": self.curriculum.state.get("current_phase", 1),
            "selected_plugins": self._last_selection_info,
            "actions": self._current_step_actions,
            "tasks": self.task_state.summary(),
            "curriculum": self.curriculum.summary(),
            "phase_transition": self.curriculum.current_phase(),
            "meta_skill": self.brain.state.get("meta_skill", 0.0),
            "lifecycle_event": self._lifecycle_event,
            "learning_policy": self.brain.get_learning_policy(),
        }
        if self._doc_curriculum_events:
            entry["doc_curriculum"] = self._doc_curriculum_events
        if guidance_msgs:
            entry["guidance"] = guidance_msgs
            entry["last_guidance"] = guidance_msgs[-1]
        external_knowledge_snippet = None
        ek = self.brain.state.get("external_knowledge", {})
        if isinstance(ek, dict) and ek:
            last_key = next(reversed(ek))
            external_knowledge_snippet = ek[last_key].get("text", "")
        graph_snippet = self._build_graph_context()
        if graph_snippet and external_knowledge_snippet:
            combined = graph_snippet + "\n" + external_knowledge_snippet
        else:
            combined = graph_snippet or external_knowledge_snippet
        reflection_text = generate_reflection(entry, combined)
        if reflection_text:
            entry["reflection"] = reflection_text
        self._append_to_diary(entry)
        self._lifecycle_event = None

    def _update_metrics(self) -> None:
        self.metrics.step()
        accepted = sum(1 for a in self._current_step_actions if a.get("result") == "accepted")
        rejected = sum(1 for a in self._current_step_actions if a.get("result") in {"rejected", "unsafe"})
        web_consults = sum(1 for a in self._current_step_actions if a.get("web_consult"))
        self.metrics.add_accepted(accepted)
        self.metrics.add_rejected(rejected)
        self.metrics.add_web_consult(web_consults)
        mastered = 0
        for tname, tinfo in self.task_state.state.items():
            if tname.startswith("_"):
                continue
            if tinfo.get("streak", 0) >= 3 and tinfo.get("last_status") == "passing":
                mastered += 1
        self.metrics.set_tasks_mastered(mastered)
        self._last_step_stats = {
            "accepted": accepted,
            "rejected": rejected,
            "web_consults": web_consults,
            "tasks_mastered": mastered,
        }

    def _update_meta_skill(self) -> None:
        total = len(self._current_step_actions)
        accepted = sum(1 for a in self._current_step_actions if a.get("result") == "accepted")
        rate = accepted / total if total else 0.0
        self.brain.update_meta_skill(rate)

    def _update_doc_mastery(self) -> None:
        if not self.doc_curriculum_enabled:
            return
        try:
            mastery, newly_mastered = compute_and_save_doc_mastery(self.doc_concepts, self.task_state)
            self.doc_mastery = mastery
            for info in newly_mastered:
                task_id = info.get("task")
                task_entry = {}
                if hasattr(self.task_state, "state"):
                    task_entry = self.task_state.state.get(task_id, {}) if task_id else {}
                plugin = task_entry.get("plugin")
                tasks_list: List[str] = []
                if task_id and plugin:
                    tasks_list.append(f"{task_id}@{plugin}")
                elif task_id:
                    tasks_list.append(task_id)
                event = {
                    "action": "mastered_concept",
                    "concept": info.get("concept"),
                    "tasks": tasks_list,
                    "passes": info.get("passes"),
                    "streak": info.get("streak"),
                }
                self._doc_curriculum_events.append(event)
        except Exception:
            # Keep the main loop running even if mastery computation fails.
            return

    def _detect_stagnation(self) -> None:
        accepted = self._last_step_stats.get("accepted", 0)
        mastered = self._last_step_stats.get("tasks_mastered", 0)
        if accepted == 0 and mastered <= self._last_mastered:
            self._stagnation_steps += 1
        else:
            self._stagnation_steps = 0
        self._last_mastered = mastered
        if self._stagnation_steps >= 10:
            self._lifecycle_event = {
                "type": "stagnation_detected",
                "steps": self._stagnation_steps,
                "action": "none",
            }
            self._stagnation_steps = 0

    def _append_to_diary(self, entry: Dict[str, Any]) -> None:
        try:
            if DIARY_FILE.exists():
                data = json.loads(DIARY_FILE.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            else:
                data = []
        except Exception:
            data = []
        data.append(entry)
        DIARY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _consult_web(self, plugin_name: str, error_type: str) -> Dict[str, str]:
        error_docs = {
            "TypeError": "https://docs.python.org/3/library/exceptions.html#TypeError",
            "IndexError": "https://docs.python.org/3/library/exceptions.html#IndexError",
            "KeyError": "https://docs.python.org/3/library/exceptions.html#KeyError",
            "AssertionError": "https://docs.python.org/3/library/exceptions.html#AssertionError",
        }
        url = error_docs.get(error_type) or "https://docs.python.org/3/library/exceptions.html"
        graph_info = self._build_graph_context(error_type=error_type, plugin_name=plugin_name, store=True)
        try:
            html = self.web_sensor.fetch_text(url)
            text = extract_plain_text(html)
            source_id = f"web:{url}"
            self.brain.store_external_knowledge(source_id, text)
            self.brain.record_last_consult(source_id, error_type)
            out = {"url": url, "status": "ok", "source_id": source_id}
            if graph_info:
                out["graph"] = graph_info
            return out
        except Exception as e:
            out = {"url": url, "status": "error", "error": str(e)}
            if graph_info:
                out["graph"] = graph_info
            return out

    def _build_graph_context(self, error_type: str | None = None, plugin_name: str | None = None, store: bool = False):
        try:
            parts = []
            if plugin_name:
                parts.append(plugin_name)
            if error_type:
                parts.append(error_type)
            query = " ".join(parts) if parts else None
            if not query:
                return None
            g_results = query_graph(query)[:3]
            if not g_results:
                return None
            texts = []
            for r in g_results:
                label = r.get("label") or ""
                source = r.get("source") or ""
                texts.append(f"{label} (from {source})")
            graph_text = " ".join(texts)[:4000]
            source_id = f"graph:{plugin_name or 'generic'}:{error_type or 'none'}"
            if store:
                self.brain.store_external_knowledge(source_id, graph_text)
            return {"source_id": source_id, "count": len(g_results), "text": graph_text}
        except Exception:
            return None
