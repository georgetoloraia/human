from __future__ import annotations

from collections import Counter
import json
import time
from pathlib import Path
from typing import Any, Dict, List


COGNITIVE_STATE_FILE = Path("manager/cognitive_state.json")


class CognitiveCore:
    """
    No-LLM conversational cognition layer.

    This is not human consciousness. It is a stateful heuristic system that mimics
    some human-like behaviors:
    - remembers recent conversation
    - tracks drives (curiosity/caution/social/confidence)
    - forms short intentions
    - answers with continuity and self-reporting
    """

    def __init__(self) -> None:
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        default = {
            "turns": [],
            "memory": {
                "short_term": [],
                "long_term": [],
            },
            "attention": {
                "working_memory_capacity": 5,
                "working_set": [],
            },
            "drives": {
                "curiosity": 0.7,
                "caution": 0.6,
                "social": 0.8,
                "confidence": 0.3,
            },
            "emotion": {
                "frustration": 0.2,
                "satisfaction": 0.2,
                "engagement": 0.7,
            },
            "identity": {
                "name": "proto-mind",
                "kind": "learning code agent",
                "limits": [
                    "not conscious",
                    "not human",
                    "no real emotions",
                    "needs tasks and feedback to improve",
                ],
            },
            "intentions": [],
            "beliefs": {
                "teacher_wants_human_like_thinking": False,
                "teacher_wants_no_llm": False,
            },
            "comprehension": {
                "last_topics": [],
                "last_intent": "unknown",
                "confidence": 0.5,
                "open_questions": [],
            },
            "learning_agenda": {
                "targets": [],
                "current_focus": None,
                "questions_asked": 0,
            },
            "lesson_mode": {
                "active": False,
                "topic": None,
                "examples": [],
                "quizzes": [],
                "pending_quiz": None,
                "history": [],
                "score": {"correct": 0, "attempted": 0},
                "last_feedback": "",
            },
            "habits": {
                "patterns": {},
                "plugins": {},
                "teacher_topics": {},
            },
            "sleep": {
                "cycles": 0,
                "last_consolidation_age": None,
                "last_summary": "",
            },
            "last_summary": "",
            "_version": "1.0",
        }
        if not COGNITIVE_STATE_FILE.exists():
            return default
        try:
            loaded = json.loads(COGNITIVE_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                # merge shallow defaults
                for k, v in default.items():
                    loaded.setdefault(k, v)
                loaded.setdefault("drives", {})
                for k, v in default["drives"].items():
                    loaded["drives"].setdefault(k, v)
                loaded.setdefault("beliefs", {})
                for k, v in default["beliefs"].items():
                    loaded["beliefs"].setdefault(k, v)
                loaded.setdefault("identity", default["identity"])
                loaded.setdefault("intentions", [])
                loaded.setdefault("turns", [])
                loaded.setdefault("memory", {})
                loaded["memory"].setdefault("short_term", [])
                loaded["memory"].setdefault("long_term", [])
                loaded.setdefault("attention", {})
                loaded["attention"].setdefault("working_memory_capacity", 5)
                loaded["attention"].setdefault("working_set", [])
                loaded.setdefault("emotion", {})
                for k, v in default["emotion"].items():
                    loaded["emotion"].setdefault(k, v)
                loaded.setdefault("habits", {})
                loaded["habits"].setdefault("patterns", {})
                loaded["habits"].setdefault("plugins", {})
                loaded["habits"].setdefault("teacher_topics", {})
                loaded.setdefault("comprehension", {})
                loaded["comprehension"].setdefault("last_topics", [])
                loaded["comprehension"].setdefault("last_intent", "unknown")
                loaded["comprehension"].setdefault("confidence", 0.5)
                loaded["comprehension"].setdefault("open_questions", [])
                loaded.setdefault("learning_agenda", {})
                loaded["learning_agenda"].setdefault("targets", [])
                loaded["learning_agenda"].setdefault("current_focus", None)
                loaded["learning_agenda"].setdefault("questions_asked", 0)
                loaded.setdefault("lesson_mode", {})
                loaded["lesson_mode"].setdefault("active", False)
                loaded["lesson_mode"].setdefault("topic", None)
                loaded["lesson_mode"].setdefault("examples", [])
                loaded["lesson_mode"].setdefault("quizzes", [])
                loaded["lesson_mode"].setdefault("pending_quiz", None)
                loaded["lesson_mode"].setdefault("history", [])
                loaded["lesson_mode"].setdefault("score", {"correct": 0, "attempted": 0})
                loaded["lesson_mode"].setdefault("last_feedback", "")
                loaded.setdefault("sleep", {})
                loaded["sleep"].setdefault("cycles", 0)
                loaded["sleep"].setdefault("last_consolidation_age", None)
                loaded["sleep"].setdefault("last_summary", "")
                return loaded
        except Exception:
            pass
        return default

    def _save(self) -> None:
        COGNITIVE_STATE_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    @staticmethod
    def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def _recent_turns(self, n: int = 6) -> List[Dict[str, Any]]:
        turns = self.state.get("turns", [])
        if not isinstance(turns, list):
            return []
        return [t for t in turns[-n:] if isinstance(t, dict)]

    def _append_short_term_memory(self, kind: str, content: Dict[str, Any]) -> None:
        mem = self.state.setdefault("memory", {})
        stm = mem.setdefault("short_term", [])
        stm.append({"ts": time.time(), "kind": kind, "content": content})
        if len(stm) > 60:
            mem["short_term"] = stm[-60:]
        self.state["memory"] = mem

    def _refresh_working_memory(self) -> None:
        attention = self.state.setdefault("attention", {})
        cap = int(attention.get("working_memory_capacity", 5) or 5)
        cap = max(2, min(12, cap))
        stm = self.state.get("memory", {}).get("short_term", [])
        habits = self.state.get("habits", {})
        teacher_topics = habits.get("teacher_topics", {}) if isinstance(habits, dict) else {}
        top_topics = [k for k, _ in sorted((teacher_topics or {}).items(), key=lambda kv: kv[1], reverse=True)[:2]]
        working: List[Dict[str, Any]] = []
        for item in (stm[-cap:] if isinstance(stm, list) else []):
            if isinstance(item, dict):
                working.append(item)
        if top_topics:
            working.append({"kind": "top_teacher_topics", "content": {"topics": top_topics}})
        intentions = self.state.get("intentions", [])
        if intentions:
            working.append({"kind": "intentions", "content": {"items": intentions[:3]}})
        attention["working_set"] = working[-cap:]
        attention["working_memory_capacity"] = cap
        self.state["attention"] = attention

    def _update_emotion_from_teacher_text(self, message: str) -> None:
        low = (message or "").lower()
        emotion = self.state.setdefault("emotion", {})
        if any(k in low for k in ("wrong", "bad", "not enough", "stop")):
            emotion["frustration"] = self._clip(float(emotion.get("frustration", 0.2)) + 0.08)
            emotion["satisfaction"] = self._clip(float(emotion.get("satisfaction", 0.2)) - 0.03)
        if any(k in low for k in ("good", "great", "nice", "thanks")):
            emotion["satisfaction"] = self._clip(float(emotion.get("satisfaction", 0.2)) + 0.07)
            emotion["frustration"] = self._clip(float(emotion.get("frustration", 0.2)) - 0.03)
        if any(k in low for k in ("talk", "chat", "human", "learn")):
            emotion["engagement"] = self._clip(float(emotion.get("engagement", 0.7)) + 0.03)
        self.state["emotion"] = emotion

    def _reinforce_habit(self, bucket: str, key: str, amount: float = 1.0) -> None:
        if not key:
            return
        habits = self.state.setdefault("habits", {})
        target = habits.setdefault(bucket, {})
        target[key] = float(target.get(key, 0.0)) + amount
        # cap size
        if len(target) > 50:
            for k, _ in sorted(target.items(), key=lambda kv: kv[1])[:-50]:
                target.pop(k, None)
        habits[bucket] = target
        self.state["habits"] = habits

    def _extract_topics(self, text: str) -> List[str]:
        low = text.lower()
        topics = []
        keywords = {
            "thinking": ["think", "thinking", "reason"],
            "learning": ["learn", "training", "improve", "grow"],
            "chat": ["talk", "chat", "speak", "conversation"],
            "safety": ["safe", "careful", "stable"],
            "tasks": ["task", "test", "plugin", "code"],
            "no_llm": ["no llm", "without llm", "offline"],
            "human_like": ["human", "natural"],
        }
        for topic, words in keywords.items():
            if any(w in low for w in words):
                topics.append(topic)
        return topics

    def _infer_intent(self, text: str, topics: List[str]) -> str:
        low = (text or "").lower().strip()
        if not low:
            return "empty"
        if "?" in low or any(low.startswith(x) for x in ("what", "how", "why", "when", "can you", "do you")):
            if "learn" in low:
                return "learning_question"
            return "question"
        if any(k in low for k in ("learn", "teach", "train", "practice")):
            return "teaching_request"
        if any(k in low for k in ("focus on", "be careful", "explore", "talk more")):
            return "behavior_instruction"
        if "human" in low and "think" in low:
            return "cognitive_goal"
        if topics:
            return "statement"
        return "unclear"

    def _update_comprehension(self, message: str, topics: List[str]) -> None:
        comp = self.state.setdefault("comprehension", {})
        intent = self._infer_intent(message, topics)
        confidence = 0.35
        if len(message.strip()) >= 6:
            confidence += 0.15
        if topics:
            confidence += 0.2
        if intent not in {"unclear", "empty"}:
            confidence += 0.15
        if len(message.split()) >= 4:
            confidence += 0.1
        confidence = self._clip(confidence)
        comp["last_topics"] = topics[:6]
        comp["last_intent"] = intent
        comp["confidence"] = confidence
        open_q = comp.get("open_questions", [])
        if not isinstance(open_q, list):
            open_q = []
        if intent in {"unclear", "teaching_request", "cognitive_goal"} and confidence < 0.75:
            q = "What should I focus on first: talking, coding tasks, or audio speech?"
            if q not in open_q:
                open_q.append(q)
        comp["open_questions"] = open_q[-8:]
        self.state["comprehension"] = comp

    def _extract_learning_targets(self, message: str) -> List[str]:
        low = (message or "").lower()
        candidates = []
        known = [
            "talking", "speech", "audio", "coding", "tasks", "python",
            "list_sum", "memory", "reasoning", "conversation", "impulses",
        ]
        for k in known:
            if k in low:
                candidates.append(k)
        if "learn" in low and not candidates:
            candidates.append("general_learning")
        return candidates

    def _update_learning_agenda(self, message: str, topics: List[str]) -> None:
        agenda = self.state.setdefault("learning_agenda", {})
        targets = agenda.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        new_targets = self._extract_learning_targets(message)
        for t in new_targets:
            if t not in targets:
                targets.append(t)
        agenda["targets"] = targets[-12:]
        if not agenda.get("current_focus") and targets:
            agenda["current_focus"] = targets[0]
        if "focus on" in (message or "").lower() and new_targets:
            agenda["current_focus"] = new_targets[0]
        self.state["learning_agenda"] = agenda

    def _maybe_followup_question(self) -> str | None:
        comp = self.state.get("comprehension", {}) or {}
        agenda = self.state.get("learning_agenda", {}) or {}
        open_q = comp.get("open_questions", []) or []
        confidence = float(comp.get("confidence", 0.5) or 0.5)
        intent = str(comp.get("last_intent", "unknown"))
        current_focus = agenda.get("current_focus")
        targets = agenda.get("targets", []) or []

        if open_q and confidence < 0.75:
            return str(open_q[-1])
        if intent in {"teaching_request", "cognitive_goal"}:
            return "What do you want me to learn first, and how will you judge if I understood it?"
        if current_focus:
            return f"Do you want me to keep learning {current_focus}, or switch to another goal?"
        if not targets:
            return "What do you want to teach me right now?"
        return None

    def _append_question_if_needed(self, base_reply: str) -> str:
        lesson = self.state.get("lesson_mode", {}) or {}
        if lesson.get("active"):
            return base_reply
        question = self._maybe_followup_question()
        if not question:
            return base_reply
        agenda = self.state.setdefault("learning_agenda", {})
        agenda["questions_asked"] = int(agenda.get("questions_asked", 0) or 0) + 1
        self.state["learning_agenda"] = agenda
        if base_reply.strip().endswith("?"):
            return base_reply
        return f"{base_reply} {question}"

    @staticmethod
    def _norm_text(text: str) -> str:
        chars = []
        for ch in (text or "").lower():
            if ch.isalnum() or ch.isspace():
                chars.append(ch)
            else:
                chars.append(" ")
        return " ".join("".join(chars).split())

    def _lesson(self) -> Dict[str, Any]:
        lesson = self.state.setdefault("lesson_mode", {})
        lesson.setdefault("active", False)
        lesson.setdefault("topic", None)
        lesson.setdefault("examples", [])
        lesson.setdefault("quizzes", [])
        lesson.setdefault("pending_quiz", None)
        lesson.setdefault("history", [])
        lesson.setdefault("score", {"correct": 0, "attempted": 0})
        lesson.setdefault("last_feedback", "")
        self.state["lesson_mode"] = lesson
        return lesson

    def _lesson_summary(self) -> str:
        lesson = self._lesson()
        score = lesson.get("score", {}) or {}
        attempted = int(score.get("attempted", 0) or 0)
        correct = int(score.get("correct", 0) or 0)
        acc = (correct / attempted) if attempted else 0.0
        return (
            f"Lesson topic={lesson.get('topic') or 'none'}, "
            f"examples={len(lesson.get('examples', []) or [])}, "
            f"quizzes={len(lesson.get('quizzes', []) or [])}, "
            f"score={correct}/{attempted} ({acc:.0%})"
        )

    def _lesson_handle(self, message: str) -> str | None:
        low = (message or "").strip().lower()
        lesson = self._lesson()

        if low in {"lesson help", "help lesson", "lesson"}:
            return (
                "Lesson mode commands: `lesson start <topic>`, `example: ...`, "
                "`quiz: <question> || <answer>`, `ask quiz`, `answer: ...`, "
                "`check understanding`, `lesson status`, `lesson stop`."
            )

        if low.startswith("lesson start ") or low.startswith("start lesson "):
            topic = message.split(" ", 2)[2].strip() if low.startswith("lesson start ") else message.split(" ", 2)[2].strip()
            if not topic:
                topic = "general"
            lesson.update(
                {
                    "active": True,
                    "topic": topic,
                    "examples": [],
                    "quizzes": [],
                    "pending_quiz": None,
                    "history": [],
                    "score": {"correct": 0, "attempted": 0},
                    "last_feedback": "",
                }
            )
            self.state["lesson_mode"] = lesson
            self._append_short_term_memory("lesson_start", {"topic": topic})
            return (
                f"Lesson mode started for topic '{topic}'. "
                "Teach me with `example:` lines, then add quizzes with `quiz: question || answer`."
            )

        if low in {"lesson stop", "end lesson", "lesson end"}:
            lesson["active"] = False
            lesson["pending_quiz"] = None
            self.state["lesson_mode"] = lesson
            return "Lesson mode stopped. " + self._lesson_summary()

        if low in {"lesson status", "check understanding", "understanding", "lesson summary"}:
            extra = ""
            if lesson.get("pending_quiz") is not None:
                idx = int(lesson.get("pending_quiz"))
                quizzes = lesson.get("quizzes", []) or []
                if 0 <= idx < len(quizzes):
                    extra = f" Pending quiz: {quizzes[idx].get('q')}"
            return self._lesson_summary() + extra

        if low.startswith("example:"):
            if not lesson.get("active"):
                return "Start lesson mode first with `lesson start <topic>`."
            ex = message.split(":", 1)[1].strip()
            if not ex:
                return "Please provide an example after `example:`."
            examples = lesson.get("examples", []) or []
            examples.append(ex)
            lesson["examples"] = examples[-20:]
            self.state["lesson_mode"] = lesson
            self._append_short_term_memory("lesson_example", {"topic": lesson.get("topic"), "example": ex})
            return f"I stored example {len(lesson['examples'])} for '{lesson.get('topic')}'. Add more or give a quiz."

        if low.startswith("quiz:"):
            if not lesson.get("active"):
                return "Start lesson mode first with `lesson start <topic>`."
            payload = message.split(":", 1)[1].strip()
            q = payload
            a = None
            for sep in ("||", "=>", "|"):
                if sep in payload:
                    left, right = payload.split(sep, 1)
                    q = left.strip()
                    a = right.strip()
                    break
            if not q:
                return "Quiz format: `quiz: <question> || <answer>`."
            quizzes = lesson.get("quizzes", []) or []
            quizzes.append({"q": q, "a": a, "added_ts": time.time()})
            lesson["quizzes"] = quizzes[-30:]
            self.state["lesson_mode"] = lesson
            self._append_short_term_memory("lesson_quiz_added", {"q": q, "has_answer": bool(a)})
            if a:
                return f"Quiz stored ({len(lesson['quizzes'])}). Ask me with `ask quiz`."
            return f"Question stored ({len(lesson['quizzes'])}), but no answer key yet. Use `quiz: q || answer` for measurable scoring."

        if low in {"ask quiz", "quiz me", "ask me a quiz"}:
            if not lesson.get("active"):
                return "Start lesson mode first with `lesson start <topic>`."
            quizzes = lesson.get("quizzes", []) or []
            if not quizzes:
                return "No quizzes stored yet. Add one with `quiz: question || answer`."
            # pick first unanswered-recent quiz or rotate
            idx = int(time.time()) % len(quizzes)
            lesson["pending_quiz"] = idx
            self.state["lesson_mode"] = lesson
            q = quizzes[idx].get("q", "")
            return f"Quiz ({idx + 1}/{len(quizzes)}): {q} Reply with `answer: ...`."

        if low.startswith("answer:"):
            if not lesson.get("active"):
                return "Start lesson mode first with `lesson start <topic>`."
            pending = lesson.get("pending_quiz")
            if pending is None:
                return "No active quiz question. Use `ask quiz` first."
            quizzes = lesson.get("quizzes", []) or []
            if not isinstance(pending, int) or not (0 <= pending < len(quizzes)):
                lesson["pending_quiz"] = None
                self.state["lesson_mode"] = lesson
                return "Pending quiz state was invalid. Use `ask quiz` again."
            user_answer = message.split(":", 1)[1].strip()
            gold = (quizzes[pending].get("a") or "").strip()
            if not gold:
                lesson["pending_quiz"] = None
                self.state["lesson_mode"] = lesson
                return "This quiz has no answer key, so I cannot score it yet."
            n_user = self._norm_text(user_answer)
            n_gold = self._norm_text(gold)
            exact = n_user == n_gold and bool(n_gold)
            user_tokens = set(n_user.split())
            gold_tokens = set(n_gold.split())
            overlap = (len(user_tokens & gold_tokens) / max(1, len(gold_tokens))) if gold_tokens else 0.0
            correct = exact or overlap >= 0.6
            score = lesson.get("score", {}) or {"correct": 0, "attempted": 0}
            score["attempted"] = int(score.get("attempted", 0) or 0) + 1
            if correct:
                score["correct"] = int(score.get("correct", 0) or 0) + 1
            lesson["score"] = score
            feedback = (
                f"{'Correct' if correct else 'Not correct yet'}. "
                f"Expected: {gold}. "
                f"Similarity={overlap:.0%}."
            )
            lesson["last_feedback"] = feedback
            history = lesson.get("history", []) or []
            history.append(
                {
                    "q": quizzes[pending].get("q"),
                    "gold": gold,
                    "answer": user_answer,
                    "correct": correct,
                    "overlap": overlap,
                    "ts": time.time(),
                }
            )
            lesson["history"] = history[-50:]
            lesson["pending_quiz"] = None
            self.state["lesson_mode"] = lesson
            self._append_short_term_memory("lesson_answer", {"correct": correct, "overlap": overlap, "topic": lesson.get("topic")})
            if correct:
                self._reinforce_habit("teacher_topics", str(lesson.get("topic") or "lesson"), amount=0.5)
            return feedback + " " + self._lesson_summary()

        return None

    def _update_beliefs_and_drives(self, message: str) -> None:
        low = message.lower()
        drives = self.state["drives"]
        beliefs = self.state["beliefs"]

        if "without llm" in low or "no llm" in low:
            beliefs["teacher_wants_no_llm"] = True
            drives["curiosity"] = self._clip(drives["curiosity"] + 0.03)
            drives["caution"] = self._clip(drives["caution"] + 0.02)
        if "human thinking" in low or "think like human" in low:
            beliefs["teacher_wants_human_like_thinking"] = True
            drives["social"] = self._clip(drives["social"] + 0.04)
        if any(k in low for k in ("good", "nice", "great", "thanks")):
            drives["confidence"] = self._clip(drives["confidence"] + 0.05)
        if any(k in low for k in ("wrong", "bad", "not enough", "no")):
            drives["confidence"] = self._clip(drives["confidence"] - 0.05)
            drives["caution"] = self._clip(drives["caution"] + 0.04)
        if any(k in low for k in ("careful", "safe", "stable")):
            drives["caution"] = self._clip(drives["caution"] + 0.08)
        if any(k in low for k in ("explore", "creative", "more")):
            drives["curiosity"] = self._clip(drives["curiosity"] + 0.06)

        self.state["drives"] = drives
        self.state["beliefs"] = beliefs

    def _set_intentions(self, topics: List[str]) -> None:
        intentions: List[str] = []
        if "human_like" in topics or self.state["beliefs"].get("teacher_wants_human_like_thinking"):
            intentions.append("simulate human-like cognition with explicit memory, goals, and reflection")
        if "no_llm" in topics or self.state["beliefs"].get("teacher_wants_no_llm"):
            intentions.append("avoid relying on LLM responses")
        if "chat" in topics:
            intentions.append("keep conversational continuity")
        if "learning" in topics:
            intentions.append("connect guidance to training behavior")
        if not intentions:
            intentions.append("understand the teacher request and respond clearly")
        self.state["intentions"] = intentions[:4]

    def _status_line(self) -> str:
        d = self.state.get("drives", {})
        style = self.style()
        return (
            f"My current internal style is {style} "
            f"(curiosity={d.get('curiosity', 0):.2f}, caution={d.get('caution', 0):.2f}, "
            f"confidence={d.get('confidence', 0):.2f})."
        )

    def style(self) -> str:
        d = self.state.get("drives", {})
        emotion = self.state.get("emotion", {})
        if float(emotion.get("frustration", 0.0)) > 0.8 and float(d.get("confidence", 0.0)) < 0.2:
            return "strained"
        if d.get("caution", 0) > d.get("curiosity", 0) + 0.1:
            return "careful"
        if d.get("curiosity", 0) > d.get("caution", 0) + 0.1:
            return "exploratory"
        return "balanced"

    def snapshot(self) -> Dict[str, Any]:
        attention = self.state.get("attention", {})
        working_set = attention.get("working_set", []) if isinstance(attention, dict) else []
        memory = self.state.get("memory", {})
        long_term = memory.get("long_term", []) if isinstance(memory, dict) else []
        habits = self.state.get("habits", {})
        return {
            "drives": dict(self.state.get("drives", {})),
            "emotion": dict(self.state.get("emotion", {})),
            "beliefs": dict(self.state.get("beliefs", {})),
            "comprehension": dict(self.state.get("comprehension", {})),
            "learning_agenda": dict(self.state.get("learning_agenda", {})),
            "lesson_mode": dict(self.state.get("lesson_mode", {})),
            "intentions": list(self.state.get("intentions", [])),
            "style": self.style(),
            "attention": {
                "working_memory_capacity": attention.get("working_memory_capacity", 5) if isinstance(attention, dict) else 5,
                "working_set_size": len(working_set) if isinstance(working_set, list) else 0,
            },
            "memory": {
                "short_term_size": len(memory.get("short_term", [])) if isinstance(memory, dict) else 0,
                "long_term_size": len(long_term) if isinstance(long_term, list) else 0,
            },
            "top_habits": {
                "patterns": [k for k, _ in sorted((habits.get("patterns", {}) or {}).items(), key=lambda kv: kv[1], reverse=True)[:3]]
                if isinstance(habits, dict)
                else [],
                "plugins": [k for k, _ in sorted((habits.get("plugins", {}) or {}).items(), key=lambda kv: kv[1], reverse=True)[:3]]
                if isinstance(habits, dict)
                else [],
            },
            "sleep": dict(self.state.get("sleep", {})),
            "last_summary": self.state.get("last_summary", ""),
        }

    def observe_teacher_signal(self, message: str) -> Dict[str, Any]:
        topics = self._extract_topics(message or "")
        self._update_beliefs_and_drives(message or "")
        self._update_emotion_from_teacher_text(message or "")
        self._update_comprehension(message or "", topics)
        self._update_learning_agenda(message or "", topics)
        self._set_intentions(topics)
        self._append_short_term_memory("teacher", {"message": message or "", "topics": topics})
        for topic in topics:
            self._reinforce_habit("teacher_topics", topic, amount=1.0)
        self._refresh_working_memory()
        self.state["last_summary"] = f"Observed teacher signal: {', '.join(topics) or 'general'}"
        self._save()
        return self.snapshot()

    def observe_outcome(
        self,
        *,
        reward: float,
        error_type: str = "OK",
        task_results: Dict[str, bool] | None = None,
        domain: str = "code",
        plugin_name: str | None = None,
        pattern_name: str | None = None,
    ) -> Dict[str, Any]:
        drives = self.state.get("drives", {})
        task_results = task_results or {}
        passed = sum(1 for ok in task_results.values() if ok)
        failed = sum(1 for ok in task_results.values() if not ok)
        success = reward > 0

        # Confidence follows recent success/failure.
        if success:
            drives["confidence"] = self._clip(float(drives.get("confidence", 0.3)) + 0.03 + min(reward, 1.0) * 0.02)
            drives["caution"] = self._clip(float(drives.get("caution", 0.6)) - 0.01)
        else:
            drives["confidence"] = self._clip(float(drives.get("confidence", 0.3)) - 0.03)
            drives["caution"] = self._clip(float(drives.get("caution", 0.6)) + 0.03)

        # Curiosity decreases slightly on repeated failure, increases on progress.
        if failed > passed:
            drives["curiosity"] = self._clip(float(drives.get("curiosity", 0.7)) - 0.02)
        elif passed > 0 or success:
            drives["curiosity"] = self._clip(float(drives.get("curiosity", 0.7)) + 0.01)

        # Certain errors encourage caution.
        if error_type in {"TypeError", "AttributeError", "Unsafe"}:
            drives["caution"] = self._clip(float(drives.get("caution", 0.6)) + 0.02)

        # Env practice success can raise confidence with lower penalty.
        if domain == "env" and success:
            drives["confidence"] = self._clip(float(drives.get("confidence", 0.3)) + 0.01)

        emotion = self.state.setdefault("emotion", {})
        if success:
            emotion["satisfaction"] = self._clip(float(emotion.get("satisfaction", 0.2)) + 0.05)
            emotion["frustration"] = self._clip(float(emotion.get("frustration", 0.2)) - 0.03)
        else:
            emotion["frustration"] = self._clip(float(emotion.get("frustration", 0.2)) + 0.06)
            emotion["satisfaction"] = self._clip(float(emotion.get("satisfaction", 0.2)) - 0.02)
        if task_results:
            progress = passed / max(1, len(task_results))
            emotion["engagement"] = self._clip(float(emotion.get("engagement", 0.7)) * 0.95 + 0.05 * progress)
        self.state["emotion"] = emotion
        self.state["drives"] = drives
        self._append_short_term_memory(
            "outcome",
            {
                "domain": domain,
                "reward": reward,
                "error_type": error_type,
                "plugin": plugin_name,
                "pattern": pattern_name,
                "passed": passed,
                "failed": failed,
            },
        )
        if reward > 0 and pattern_name:
            self._reinforce_habit("patterns", pattern_name, amount=1.5)
        elif pattern_name:
            self._reinforce_habit("patterns", pattern_name, amount=0.2)
        if plugin_name:
            self._reinforce_habit("plugins", plugin_name, amount=1.0 if reward > 0 else 0.1)
        self._refresh_working_memory()
        summary_bits = [
            f"outcome domain={domain}",
            f"reward={reward:.3f}",
            f"error={error_type}",
        ]
        if plugin_name:
            summary_bits.append(f"plugin={plugin_name}")
        if pattern_name:
            summary_bits.append(f"pattern={pattern_name}")
        if task_results:
            summary_bits.append(f"tasks={passed} pass/{failed} fail")
        self.state["last_summary"] = "Observed " + ", ".join(summary_bits)
        self._save()
        return self.snapshot()

    def observe_step(self, *, age: int | None = None, domain: str | None = None, had_actions: bool = False) -> Dict[str, Any]:
        # Homeostasis: drift drives/emotions toward baselines each step.
        baselines = {"curiosity": 0.7, "caution": 0.6, "social": 0.8, "confidence": 0.3}
        drives = self.state.setdefault("drives", {})
        for key, base in baselines.items():
            current = float(drives.get(key, base))
            drives[key] = self._clip(current * 0.98 + base * 0.02)
        self.state["drives"] = drives

        emotion = self.state.setdefault("emotion", {})
        emotion_baselines = {"frustration": 0.2, "satisfaction": 0.2, "engagement": 0.7}
        for key, base in emotion_baselines.items():
            current = float(emotion.get(key, base))
            decay = 0.03 if key != "engagement" else 0.02
            emotion[key] = self._clip(current * (1 - decay) + base * decay)
        if had_actions:
            emotion["engagement"] = self._clip(float(emotion.get("engagement", 0.7)) + 0.01)
        self.state["emotion"] = emotion

        if domain:
            self._append_short_term_memory("step", {"age": age, "domain": domain, "had_actions": had_actions})
        self._refresh_working_memory()
        self._save()
        return self.snapshot()

    def maybe_sleep_consolidate(self, *, age: int | None = None, force: bool = False) -> Dict[str, Any] | None:
        sleep = self.state.setdefault("sleep", {})
        last_age = sleep.get("last_consolidation_age")
        if not force:
            if age is None:
                return None
            if last_age is not None and isinstance(last_age, int) and age - last_age < 10:
                return None
            if age < 10:
                return None

        mem = self.state.setdefault("memory", {})
        stm = mem.get("short_term", [])
        if not isinstance(stm, list) or not stm:
            return None
        recent = [x for x in stm[-30:] if isinstance(x, dict)]
        if not recent:
            return None

        topic_counter: Counter[str] = Counter()
        pattern_counter: Counter[str] = Counter()
        error_counter: Counter[str] = Counter()
        for item in recent:
            kind = item.get("kind")
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
            if kind == "teacher":
                for t in content.get("topics", []) or []:
                    topic_counter[str(t)] += 1
            if kind == "outcome":
                pattern = content.get("pattern")
                if pattern:
                    pattern_counter[str(pattern)] += 1
                err = content.get("error_type")
                if err and err != "OK":
                    error_counter[str(err)] += 1

        summary = {
            "ts": time.time(),
            "age": age,
            "top_teacher_topics": [k for k, _ in topic_counter.most_common(3)],
            "top_patterns": [k for k, _ in pattern_counter.most_common(3)],
            "top_errors": [k for k, _ in error_counter.most_common(3)],
            "intentions": list(self.state.get("intentions", []))[:3],
        }
        ltm = mem.setdefault("long_term", [])
        ltm.append(summary)
        if len(ltm) > 100:
            mem["long_term"] = ltm[-100:]
        # Keep some recency but clear older STM items to simulate consolidation.
        mem["short_term"] = stm[-20:]
        self.state["memory"] = mem

        for topic, count in topic_counter.items():
            self._reinforce_habit("teacher_topics", topic, amount=float(count) * 0.5)
        for pattern, count in pattern_counter.items():
            self._reinforce_habit("patterns", pattern, amount=float(count) * 0.2)

        sleep["cycles"] = int(sleep.get("cycles", 0) or 0) + 1
        sleep["last_consolidation_age"] = age
        sleep["last_summary"] = (
            f"topics={summary['top_teacher_topics']} patterns={summary['top_patterns']} errors={summary['top_errors']}"
        )
        self.state["sleep"] = sleep
        self._refresh_working_memory()
        self._save()
        return summary

    def _build_reply(self, message: str, external: Dict[str, Any] | None = None) -> str:
        external = external or {}
        low = message.lower()
        topics = self._extract_topics(message)
        recent = self._recent_turns()
        self._set_intentions(topics)
        self._update_comprehension(message, topics)
        self._update_learning_agenda(message, topics)
        lesson_reply = self._lesson_handle(message)
        if lesson_reply is not None:
            return lesson_reply

        age = external.get("age", 0)
        skill = external.get("skill", 0)
        meta_skill = external.get("meta_skill", 0.0)
        last_reflection = external.get("last_reflection")

        if "human thinking" in low and ("without llm" in low or "no llm" in low):
            parts = [
                "I cannot become a real human mind, but I can use a no-LLM cognitive architecture that behaves more human-like.",
                "That means persistent memory, explicit goals, attention/focus, self-reflection, and adaptive conversation rules instead of text generation from an LLM.",
                self._status_line(),
                f"My current training state is age={age}, skill={skill}, meta_skill={meta_skill:.2f}.",
            ]
            if last_reflection:
                parts.append(f"My latest self-report was: {last_reflection[:220]}")
            parts.append("If you want, keep giving teacher instructions and I will treat them as priorities for my behavior.")
            return self._append_question_if_needed(" ".join(parts))

        if any(k in low for k in ("what are you doing", "status")):
            wm = self.state.get("attention", {}).get("working_set", [])
            reply = [
                f"I am maintaining conversational memory and intentions ({'; '.join(self.state.get('intentions', []))}).",
                f"Training state: age={age}, skill={skill}, meta_skill={meta_skill:.2f}.",
                self._status_line(),
                f"My working memory is holding {len(wm) if isinstance(wm, list) else 0} active items.",
            ]
            if recent:
                prev = recent[-1].get("user", "")
                if prev:
                    reply.append(f"I remember your recent focus: '{prev[:100]}'.")
            return self._append_question_if_needed(" ".join(reply))

        if any(k in low for k in ("how do you think", "how are you thinking", "reason")):
            steps = [
                "I parse your message into topics and intent",
                "I update drives and beliefs",
                "I form short intentions",
                "I read recent conversation memory",
                "I generate a rule-based response linked to current training state",
            ]
            return self._append_question_if_needed(
                "I use explicit steps, not an LLM: " + "; ".join(steps) + ". " + self._status_line()
            )

        if any(k in low for k in ("who are you", "what are you")):
            return self._append_question_if_needed(
                (
                "I am a no-LLM cognitive agent layer on top of your training system. "
                "I use memory, heuristics, and internal drives to respond consistently, but I am not conscious."
                )
            )

        if any(k in low for k in ("learn", "improve", "grow")):
            return self._append_question_if_needed(
                (
                "I learn by turning your guidance into priorities, then the training loop tests code changes and updates memory from pass/fail outcomes. "
                + self._status_line()
                )
            )

        return self._append_question_if_needed(
            (
            "I recorded your message and updated my internal drives and intentions. "
            f"{self._status_line()} "
            "I can continue with no-LLM human-like behavior simulation using short/long-term memory, habits, and self-reporting."
            )
        )

    def respond(self, message: str, external: Dict[str, Any] | None = None) -> str:
        self._update_beliefs_and_drives(message)
        reply = self._build_reply(message, external=external)
        turns = self.state.get("turns", [])
        turns.append(
            {
                "ts": time.time(),
                "user": message,
                "assistant": reply,
                "topics": self._extract_topics(message),
                "intentions": list(self.state.get("intentions", [])),
            }
        )
        if len(turns) > 200:
            turns = turns[-200:]
        self.state["turns"] = turns
        self._append_short_term_memory("dialogue_turn", {"user": message, "assistant": reply[:200]})
        self._refresh_working_memory()
        self.state["last_summary"] = f"Last topic(s): {', '.join(self._extract_topics(message)) or 'general'}"
        self._save()
        return reply
