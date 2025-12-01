import json
from pathlib import Path
from typing import Dict, List

STATE_FILE = Path("manager/curriculum_state.json")


class Curriculum:
    def __init__(self, required_streak: int = 3):
        self.required_streak = required_streak
        self.state = {
            "current_phase": 1,
            "task_status": {},  # task -> {"status": locked|unlocked|mastered, "phase": int, "streak": int}
            "phase_stats": {},  # phase -> {"tasks": [names], "unlocked": bool, "mastered": bool}
        }
        if STATE_FILE.exists():
            try:
                self.state = json.loads(STATE_FILE.read_text())
            except Exception:
                pass
        self._normalize_state()

    def save(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _normalize_state(self):
        phase_stats = {}
        for k, v in self.state.get("phase_stats", {}).items():
            try:
                ik = int(k)
            except Exception:
                continue
            phase_stats[ik] = v
        self.state["phase_stats"] = phase_stats
        for name, info in self.state.get("task_status", {}).items():
            try:
                info["phase"] = int(info.get("phase", 1))
            except Exception:
                info["phase"] = 1
            self.state["task_status"][name] = info

    def sync_tasks(self, tasks: List[Dict]):
        # ensure phase stats exists
        for t in tasks:
            phase = int(t.get("phase", 1))
            name = t.get("name")
            if not name:
                continue
            plugin = t.get("target_plugin")
            self.state["phase_stats"].setdefault(
                phase, {"tasks": [], "unlocked": phase == 1, "mastered": False}
            )
            if name not in self.state["phase_stats"][phase]["tasks"]:
                self.state["phase_stats"][phase]["tasks"].append(name)
            status = self.state["task_status"].get(
                name,
                {
                    "status": "unlocked" if phase == 1 else "locked",
                    "phase": phase,
                    "streak": 0,
                    "plugin": plugin,
                    "prerequisites": t.get("prerequisites", []),
                },
            )
            status["phase"] = phase
            status["plugin"] = plugin
            status["prerequisites"] = t.get("prerequisites", [])
            self.state["task_status"][name] = status
        self._recompute_unlocks()
        self.save()

    def _recompute_unlocks(self):
        # phase 1 unlocked
        self.state["phase_stats"].setdefault(1, {"tasks": [], "unlocked": True, "mastered": False})
        for phase, info in self.state["phase_stats"].items():
            if phase == 1:
                info["unlocked"] = True
        # unlock next phases if previous mastered and prereqs met
        phases = sorted(self.state["phase_stats"].keys())
        mastered_phases = {p for p, info in self.state["phase_stats"].items() if info.get("mastered")}
        for phase in phases:
            if phase == 1:
                continue
            prev = phase - 1
            prev_mastered = self.state["phase_stats"].get(prev, {}).get("mastered", False)
            if prev_mastered:
                # ensure prerequisites of tasks in this phase are mastered
                prereq_ok = True
                for t in self.state["phase_stats"][phase]["tasks"]:
                    task_info = self.state["task_status"].get(t, {})
                    for pre in task_info.get("prerequisites", []):
                        pre_status = self.state["task_status"].get(pre, {}).get("status")
                        if pre_status != "mastered":
                            prereq_ok = False
                            break
                    if not prereq_ok:
                        break
                if prereq_ok:
                    self.state["phase_stats"][phase]["unlocked"] = True
        # set current_phase to lowest unlocked not mastered
        unlocked_phases = [p for p, info in self.state["phase_stats"].items() if info.get("unlocked")]
        if unlocked_phases:
            self.state["current_phase"] = min(
                [p for p in unlocked_phases if not self.state["phase_stats"].get(p, {}).get("mastered")]
                or [max(unlocked_phases)]
            )
        # unlock tasks in unlocked phases
        for name, info in self.state["task_status"].items():
            phase = info.get("phase", 1)
            if self.state["phase_stats"].get(phase, {}).get("unlocked"):
                if info["status"] == "locked":
                    info["status"] = "unlocked"
        self.save()

    def active_tasks(self):
        return {
            name: info
            for name, info in self.state["task_status"].items()
            if info.get("status") in {"unlocked", "mastered"}
        }

    def get_task_info(self, name: str) -> Dict:
        return self.state["task_status"].get(name, {})

    def tasks_by_plugin(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for name, info in self.active_tasks().items():
            plugin = info.get("plugin")
            if plugin:
                mapping.setdefault(plugin, []).append(name)
        return mapping

    def update_results(self, task_pass_fail: Dict[str, bool]):
        for task, success in task_pass_fail.items():
            info = self.state["task_status"].get(task)
            if not info:
                continue
            if success:
                info["streak"] = info.get("streak", 0) + 1
                if info["streak"] >= self.required_streak:
                    info["status"] = "mastered"
            else:
                info["streak"] = 0
                if info["status"] == "mastered":
                    info["status"] = "unlocked"
            self.state["task_status"][task] = info
        self._update_phase_mastery()
        self._recompute_unlocks()
        self.save()

    def _update_phase_mastery(self):
        for phase, info in self.state["phase_stats"].items():
            tasks = info.get("tasks", [])
            mastered = all(self.state["task_status"].get(t, {}).get("status") == "mastered" for t in tasks) if tasks else False
            info["mastered"] = mastered
            self.state["phase_stats"][phase] = info
        self.save()

    def summary(self) -> Dict[int, Dict[str, Any]]:
        summary: Dict[int, Dict[str, Any]] = {}
        for phase, info in self.state.get("phase_stats", {}).items():
            tasks = info.get("tasks", [])
            mastered = [t for t in tasks if self.state["task_status"].get(t, {}).get("status") == "mastered"]
            summary[phase] = {
                "tasks_total": len(tasks),
                "tasks_mastered": len(mastered),
                "unlocked": info.get("unlocked", False),
                "mastered_phase": info.get("mastered", False),
            }
        return summary

    def tasks_for_plugin(self, plugin: str) -> List[str]:
        return [
            name
            for name, info in self.state["task_status"].items()
            if info.get("plugin") == plugin and info.get("status") in {"unlocked", "mastered"}
        ]
