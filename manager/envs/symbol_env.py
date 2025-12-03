from __future__ import annotations

import random
from typing import Dict, Tuple


class SymbolEnv:
    """
    Tiny numeric pattern environment: predict next number in an arithmetic sequence.
    """

    def __init__(self) -> None:
        self.state: Dict[str, int] = {}
        self.done = False

    def reset(self) -> Dict[str, int]:
        start = random.randint(1, 5)
        step = random.choice([1, 2, 3])
        length = 3
        seq = [start + i * step for i in range(length)]
        self.state = {"sequence": seq, "step": step}
        self.done = False
        return dict(self.state)

    def step(self, action: int) -> Tuple[Dict[str, int], float, bool, Dict[str, int]]:
        if self.done:
            return dict(self.state), 0.0, True, {}
        expected = self.state["sequence"][-1] + self.state["step"]
        reward = 1.0 if action == expected else -0.5
        self.done = True
        info = {"expected": expected, "descriptor": self.describe_state()}
        return dict(self.state), reward, self.done, info

    def describe_state(self) -> Dict[str, int]:
        return {
            "env": "SymbolEnv",
            "sequence_head": self.state.get("sequence", [])[-1] if self.state.get("sequence") else None,
            "step": self.state.get("step"),
        }
