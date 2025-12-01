from pathlib import Path
import json
from typing import Any, Dict

MEMORY_FILE = Path("manager/brain_memory.json")
MEMORY_VERSION = "1.0"


class BrainMemory:
    def __init__(self):
        self.state = {
            "observations": [],
            "concepts": {},
            "attempts": {},
            "patterns": {},
            "pattern_error_stats": {},
            "external_knowledge": {},
            "error_streaks": {},
            "last_consult": {"source_id": None, "error_type": None},
            "age": 0,
            "_version": MEMORY_VERSION,
            "meta_skill": 0.0,
            "meta_skill_history": [],
        }
        if MEMORY_FILE.exists():
            try:
                self.state = json.loads(MEMORY_FILE.read_text())
            except json.JSONDecodeError:
                # If the file is corrupt, reset to defaults.
                pass
        self.state.setdefault("observations", [])
        self.state.setdefault("concepts", {})
        self.state.setdefault("attempts", {})
        self.state.setdefault("patterns", {})
        self.state.setdefault("pattern_error_stats", {})
        self.state.setdefault("external_knowledge", {})
        self.state.setdefault("error_streaks", {})
        self.state.setdefault("last_consult", {"source_id": None, "error_type": None})
        self.state.setdefault("age", 0)
        self.state.setdefault("meta_skill", 0.0)
        self.state.setdefault("meta_skill_history", [])
        self.state.setdefault("_version", MEMORY_VERSION)

    def observe(self, text: str):
        self.state["observations"].append(text)
        if len(self.state["observations"]) > 1000:
            self.state["observations"] = self.state["observations"][-1000:]
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

    def store_external_knowledge(self, source: str, text: str) -> None:
        self.state.setdefault("external_knowledge", {})
        self.state["external_knowledge"][source] = {"text": text[:4000]}
        # cap entries to avoid unbounded growth
        if len(self.state["external_knowledge"]) > 200:
            # keep most recent 200 by insertion order (Py3.7+ dict preserves)
            keys = list(self.state["external_knowledge"].keys())
            for k in keys[:-200]:
                self.state["external_knowledge"].pop(k, None)
        self._save()

    def record_error_event(self, plugin_name: str, error_type: str, success: bool) -> None:
        self.state.setdefault("error_streaks", {})
        key = f"{plugin_name}|{error_type}"
        if success:
            self.state["error_streaks"][key] = 0
        else:
            current = self.state["error_streaks"].get(key, 0)
            self.state["error_streaks"][key] = current + 1
        self._save()

    def get_error_streak(self, plugin_name: str, error_type: str) -> int:
        key = f"{plugin_name}|{error_type}"
        return int(self.state.get("error_streaks", {}).get(key, 0))

    def record_last_consult(self, source_id: str, error_type: str) -> None:
        self.state.setdefault("last_consult", {})
        self.state["last_consult"]["source_id"] = source_id
        self.state["last_consult"]["error_type"] = error_type
        self._save()

    def get_last_consult(self) -> Dict[str, Any]:
        return self.state.get("last_consult", {})

    def update_meta_skill(self, acceptance_rate: float) -> None:
        # exponential moving average
        current = float(self.state.get("meta_skill", 0.0))
        updated = 0.8 * current + 0.2 * acceptance_rate
        self.state["meta_skill"] = updated
        history = self.state.get("meta_skill_history", [])
        history.append({"rate": acceptance_rate, "score": updated})
        if len(history) > 200:
            history = history[-200:]
        self.state["meta_skill_history"] = history
        self._save()

    def _save(self):
        MEMORY_FILE.write_text(json.dumps(self.state, indent=2))
