from __future__ import annotations

from pathlib import Path
from typing import List

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

PLUGINS_DIR = Path("plugins")


class Mind:
    """
    Minimal orchestrator that:
      - perceives plugins
      - maintains models (brain, concept graph, curriculum)
      - chooses targets (via Goals)
      - applies mutations and learns from feedback
    """

    def __init__(self) -> None:
        self.brain = BrainMemory()
        self.graph = ConceptGraph()
        self.curriculum = Curriculum()
        self.goals = Goals(self.graph, self.curriculum)
        self.last_error_type: str | None = None

    def step(self) -> None:
        self.brain.grow()
        tasks = regenerate_task_tests(self.curriculum)
        self.graph.register_tasks(tasks)
        active_task_names = [t["name"] for t in tasks if "name" in t]

        self._perceive()

        target_plugins = self.goals.choose_plugins_to_grow(max_plugins=2)
        if not target_plugins:
            target_plugins = [p.name for p in PLUGINS_DIR.glob("*.py") if p.name != "__init__.py"]

        pattern_scores = self.brain.pattern_scores()
        error_scores = self.brain.pattern_error_scores(self.last_error_type) if self.last_error_type else {}

        for plugin_name in target_plugins:
            path = PLUGINS_DIR / plugin_name
            if not path.exists():
                continue
            src = path.read_text(encoding="utf-8")
            proposals = propose_mutations(src, pattern_scores, error_scores)
            if not proposals:
                continue
            backup = src
            for new_code, pattern_name in proposals:
                mutation_id = f"{plugin_name}:{pattern_name}:{hash(new_code)}"
                if not is_safe(new_code):
                    self.brain.record_attempt(mutation_id, False)
                    self.brain.record_pattern_result(pattern_name, False)
                    self.brain.record_pattern_error_result(pattern_name, "Other", False)
                    self.graph.record_test_result(plugin_name, False)
                    continue

                path.write_text(new_code, encoding="utf-8")
                tests_ok, err_type, failing_tasks = run_tests()
                eval_ok = tests_ok and evaluate(backup, new_code)

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

                if eval_ok:
                    self.brain.record_attempt(mutation_id, True)
                    self.brain.record_pattern_result(pattern_name, True)
                    self.brain.record_pattern_error_result(pattern_name, err_type, True)
                    self.graph.record_growth(plugin_name)
                    if not tests_ok:
                        self.last_error_type = err_type
                    break
                else:
                    path.write_text(backup, encoding="utf-8")
                    self.brain.record_attempt(mutation_id, False)
                    self.brain.record_pattern_result(pattern_name, False)
                    self.brain.record_pattern_error_result(pattern_name, err_type, False)
                    self.last_error_type = err_type

    def _perceive(self) -> None:
        observations = observe_codebase()
        for obs in observations:
            self.brain.observe(str(obs))
            fname = obs.get("file")
            if fname:
                p = PLUGINS_DIR / fname
                if p.exists():
                    self.graph.observe_plugin(p)
