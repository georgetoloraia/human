from pathlib import Path
import json

MEMORY_FILE = Path("manager/brain_memory.json")


class BrainMemory:
    def __init__(self):
        self.state = {
            "observations": [],
            "concepts": {},
            "attempts": {},
            "age": 0,
        }
        if MEMORY_FILE.exists():
            try:
                self.state = json.loads(MEMORY_FILE.read_text())
            except json.JSONDecodeError:
                # If the file is corrupt, reset to defaults.
                pass

    def observe(self, text: str):
        self.state["observations"].append(text)
        self._save()

    def learn_concept(self, name: str, confidence: float):
        self.state["concepts"][name] = confidence
        self._save()

    def record_attempt(self, mutation: str, success: bool):
        self.state["attempts"].setdefault(mutation, {"ok": 0, "fail": 0})
        if success:
            self.state["attempts"][mutation]["ok"] += 1
        else:
            self.state["attempts"][mutation]["fail"] += 1
        self._save()

    def grow(self):
        self.state["age"] += 1
        self._save()

    def get_skill_level(self):
        total = sum(v["ok"] for v in self.state["attempts"].values())
        return total + self.state["age"]

    def _save(self):
        MEMORY_FILE.write_text(json.dumps(self.state, indent=2))
