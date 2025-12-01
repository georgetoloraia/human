from pathlib import Path
import json

MEMORY_FILE = Path("manager/brain_memory.json")


class BrainMemory:
    def __init__(self):
        self.state = {
            "observations": [],
            "concepts": {},
            "attempts": {},
            "patterns": {},
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

    def record_pattern_result(self, pattern: str, success: bool):
        stats = self.state["patterns"].setdefault(pattern, {"ok": 0, "fail": 0})
        if success:
            stats["ok"] += 1
        else:
            stats["fail"] += 1
        self.state["patterns"][pattern] = stats
        self._save()

    def record_pattern_error_result(self, pattern: str, error_type: str, success: bool):
        per_pattern = self.state.setdefault("pattern_error_stats", {})
        pattern_stats = per_pattern.setdefault(pattern, {})
        err_stats = pattern_stats.setdefault(error_type, {"ok": 0, "fail": 0})
        if success:
            err_stats["ok"] += 1
        else:
            err_stats["fail"] += 1
        pattern_stats[error_type] = err_stats
        per_pattern[pattern] = pattern_stats
        self.state["pattern_error_stats"] = per_pattern
        self._save()

    def pattern_scores(self):
        # simple success rate heuristic with Laplace smoothing
        scores = {}
        for name, stats in self.state["patterns"].items():
            ok = stats.get("ok", 0)
            fail = stats.get("fail", 0)
            total = ok + fail
            scores[name] = (ok + 1) / (total + 2)
        return scores

    def pattern_error_scores(self, error_type: str):
        per_pattern = self.state.get("pattern_error_stats", {})
        scores = {}
        for name, err_map in per_pattern.items():
            stats = err_map.get(error_type, {"ok": 0, "fail": 0})
            ok = stats.get("ok", 0)
            fail = stats.get("fail", 0)
            total = ok + fail
            scores[name] = (ok + 1) / (total + 2)
        return scores

    def _save(self):
        MEMORY_FILE.write_text(json.dumps(self.state, indent=2))
