from __future__ import annotations

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
        self.stage: int = 0
        self.last_error_type: str | None = None
        self._last_selection_info: List[Dict[str, Any]] = []
        self._current_step_actions: List[Dict[str, Any]] = []

    def step(self) -> None:
        self._last_selection_info = []
        self._current_step_actions = []
        self._age_and_stage()
        tasks = regenerate_task_tests(self.curriculum)
        self.graph.register_tasks(tasks)
        active_task_names = [t["name"] for t in tasks if "name" in t]
        self._perceive_world()
        self._act_and_learn(active_task_names)
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
            + w_t * task_drive
        )
        return {
            "curiosity": curiosity,
            "mastery": mastery,
            "stability_penalty": stability_penalty,
            "task_count": task_count,
            "task_avg_streak": avg_streak,
            "task_failing_count": failing_count,
            "task_drive": task_drive,
            "total": total,
        }

    def _select_targets(self) -> List[str]:
        base_targets = self.goals.choose_plugins_to_grow(max_plugins=4)
        if not base_targets:
            base_targets = [p.name for p in PLUGINS_DIR.glob("*.py") if p.name != "__init__.py"]
        scored: List[Tuple[float, str, Dict[str, float]]] = []
        for name in base_targets:
            scores = self._score_plugin(name)
            scored.append((scores["total"], name, scores))
        scored.sort(reverse=True, key=lambda x: x[0])
        if self.stage == 0:
            max_plugins = 3
        elif self.stage == 1:
            max_plugins = 2
        else:
            max_plugins = 1
        selected = scored[:max_plugins]
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
        if self.stage == 0:
            max_candidates_per_plugin = 5
        elif self.stage == 1:
            max_candidates_per_plugin = 3
        elif self.stage == 2:
            max_candidates_per_plugin = 2
        else:
            max_candidates_per_plugin = 1

        for plugin_name in targets:
            path = PLUGINS_DIR / plugin_name
            if not path.exists():
                continue
            src = path.read_text(encoding="utf-8")
            error_scores = self.brain.pattern_error_scores(self.last_error_type) if self.last_error_type else {}
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

        tests_ok, err_type, failing_tasks = run_tests()
        eval_ok = tests_ok and evaluate(orig, new_code)

        mutation_id = f"{plugin_name}:{pattern_name}"

        self.graph.record_test_result(plugin_name, tests_ok)
        task_results = {}
        for tname in active_task_names:
            if tname in failing_tasks:
                task_results[tname] = False
                self.graph.record_task_result(tname, False)
            else:
                task_results[tname] = True
                self.graph.record_task_result(tname, True)
        self.curriculum.update_results(task_results)
        self.task_state.record_plugin_result(plugin_name, success=eval_ok, error_type=err_type)
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
        entry: Dict[str, Any] = {
            "age": age,
            "stage": self.stage,
            "skill": skill,
            "selected_plugins": self._last_selection_info,
            "actions": self._current_step_actions,
            "tasks": self.task_state.summary(),
        }
        external_knowledge_snippet = None
        ek = self.brain.state.get("external_knowledge", {})
        if isinstance(ek, dict) and ek:
            last_key = next(reversed(ek))
            external_knowledge_snippet = ek[last_key].get("text", "")
        reflection_text = generate_reflection(entry, external_knowledge_snippet)
        if reflection_text:
            entry["reflection"] = reflection_text
        self._append_to_diary(entry)

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
        try:
            html = self.web_sensor.fetch_text(url)
            text = extract_plain_text(html)
            source_id = f"web:{url}"
            self.brain.store_external_knowledge(source_id, text)
            return {"url": url, "status": "ok", "source_id": source_id}
        except Exception as e:
            return {"url": url, "status": "error", "error": str(e)}
