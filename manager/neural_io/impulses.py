from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class Impulse:
    t: float
    ch: int
    v: float = 1.0
    kind: str = "spike"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {"t": float(self.t), "ch": int(self.ch), "v": float(self.v), "kind": self.kind}
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Impulse":
        return cls(
            t=float(data.get("t", 0.0)),
            ch=int(data.get("ch", 0)),
            v=float(data.get("v", 1.0)),
            kind=str(data.get("kind", "spike")),
            meta=dict(data.get("meta", {}) or {}),
        )


class ImpulseStream:
    def __init__(self, impulses: Iterable[Impulse] | None = None):
        self.impulses: List[Impulse] = list(impulses or [])

    def add(self, impulse: Impulse) -> None:
        self.impulses.append(impulse)

    def extend(self, impulses: Iterable[Impulse]) -> None:
        self.impulses.extend(impulses)

    def sort(self) -> None:
        self.impulses.sort(key=lambda x: (x.t, x.ch))

    def to_list(self) -> List[Dict[str, Any]]:
        return [i.to_dict() for i in self.impulses]

    @classmethod
    def from_list(cls, items: Iterable[Dict[str, Any]]) -> "ImpulseStream":
        return cls(Impulse.from_dict(i) for i in items)

    def save_jsonl(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for impulse in self.impulses:
                f.write(json.dumps(impulse.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "ImpulseStream":
        p = Path(path)
        if not p.exists():
            return cls()
        items: List[Impulse] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(Impulse.from_dict(json.loads(line)))
                except Exception:
                    continue
        return cls(items)

    def summary(self) -> Dict[str, Any]:
        if not self.impulses:
            return {"count": 0, "channels": 0, "duration_s": 0.0}
        channels = {i.ch for i in self.impulses}
        t0 = min(i.t for i in self.impulses)
        t1 = max(i.t for i in self.impulses)
        return {
            "count": len(self.impulses),
            "channels": len(channels),
            "min_channel": min(channels),
            "max_channel": max(channels),
            "duration_s": max(0.0, t1 - t0),
        }
