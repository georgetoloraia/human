import time
from pathlib import Path

from manager.memory import BrainMemory
from manager.generator import propose_mutations
from manager.safety import is_safe
from manager.tester import run_tests
from manager.evaluator import evaluate
from manager.perception import observe_codebase
from manager.concepts import ConceptGraph
from manager.goals import Goals

PLUGINS = Path("plugins")

brain = BrainMemory()
graph = ConceptGraph()
goals = Goals(graph)


def life_cycle():
    brain.grow()

    observations = observe_codebase()
    for obs in observations:
        brain.observe(str(obs))
        fname = obs.get("file")
        if fname:
            p = PLUGINS / fname
            if p.exists():
                graph.observe_plugin(p)

    target_plugins = goals.choose_plugins_to_grow(max_plugins=2)
    if not target_plugins:
        target_plugins = [p.name for p in PLUGINS.glob("*.py")]

    for plugin_name in target_plugins:
        path = PLUGINS / plugin_name
        if not path.exists():
            continue

        src = path.read_text(encoding="utf-8")
        pattern_scores = brain.pattern_scores()
        proposals = propose_mutations(src, pattern_scores)
        if not proposals:
            continue

        backup = src
        for new_code, pattern_name in proposals:
            mutation_id = f"{plugin_name}:{pattern_name}:{hash(new_code)}"

            if not is_safe(new_code):
                brain.record_attempt(mutation_id, False)
                brain.record_pattern_result(pattern_name, False)
                graph.record_test_result(plugin_name, False)
                continue

            path.write_text(new_code, encoding="utf-8")

            tests_ok, err_type = run_tests()
            eval_ok = tests_ok and evaluate(backup, new_code)

            graph.record_test_result(plugin_name, tests_ok)

            if eval_ok:
                brain.record_attempt(mutation_id, True)
                brain.record_pattern_result(pattern_name, True)
                graph.record_growth(plugin_name)
                break  # accept one success per plugin per cycle
            else:
                path.write_text(backup, encoding="utf-8")
                brain.record_attempt(mutation_id, False)
                brain.record_pattern_result(pattern_name, False)


def run_forever():
    while True:
        print("[LIFE] age=", brain.state["age"], "skill=", brain.get_skill_level())
        life_cycle()
        time.sleep(5)


if __name__ == "__main__":
    run_forever()
