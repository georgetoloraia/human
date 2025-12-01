import time
from pathlib import Path

from manager.memory import BrainMemory
from manager.generator import grow_code
from manager.safety import is_safe
from manager.tester import run_tests
from manager.evaluator import evaluate
from manager.perception import observe_codebase

PLUGINS = Path("plugins")
brain = BrainMemory()


def life_cycle():
    brain.grow()

    observations = observe_codebase()
    for obs in observations:
        brain.observe(str(obs))

    if not PLUGINS.exists():
        return

    for p in PLUGINS.glob("*.py"):
        src = p.read_text()
        new_code = grow_code(src)

        if not new_code:
            continue

        mutation_id = f"{p.name}:{hash(new_code)}"
        if not is_safe(new_code):
            brain.record_attempt(mutation_id, False)
            continue

        backup = src
        p.write_text(new_code)

        if run_tests() and evaluate(src, new_code):
            brain.record_attempt(mutation_id, True)
        else:
            p.write_text(backup)
            brain.record_attempt(mutation_id, False)


def run_forever():
    while True:
        print("Brain age:", brain.state["age"], " Skill:", brain.get_skill_level())
        life_cycle()
        time.sleep(5)


if __name__ == "__main__":
    run_forever()
