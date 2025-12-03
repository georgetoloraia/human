from __future__ import annotations

import random
from typing import Dict, Tuple


class TextPuzzleEnv:
    """
    Simple environment that asks the agent to transform a word (reverse or uppercase).
    """

    def __init__(self) -> None:
        self.state: Dict[str, str] = {}
        self.done = False

    def reset(self) -> Dict[str, str]:
        word = random.choice(["brain", "neuron", "code", "learn"])
        target = random.choice(["reverse", "upper"])
        self.state = {"word": word, "target": target}
        self.done = False
        return dict(self.state)

    def step(self, action: str) -> Tuple[Dict[str, str], float, bool, Dict[str, str]]:
        if self.done:
            return dict(self.state), 0.0, True, {}
        reward = -0.1
        expected = ""
        if self.state.get("target") == "reverse":
            expected = self.state.get("word", "")[::-1]
        elif self.state.get("target") == "upper":
            expected = self.state.get("word", "").upper()
        if action == expected:
            reward = 1.0
            self.done = True
        next_state = dict(self.state)
        info = {"expected": expected, "descriptor": self.describe_state()}
        return next_state, reward, self.done, info

    def describe_state(self) -> Dict[str, str]:
        return {
            "env": "TextPuzzleEnv",
            "word": self.state.get("word", ""),
            "target": self.state.get("target", ""),
        }
