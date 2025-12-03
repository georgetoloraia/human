from __future__ import annotations

import asyncio
from typing import Callable, Iterable, List, Any


class AsyncOrchestrator:
    """
    Minimal async wrapper to run independent tasks in parallel when enabled.
    Falls back to synchronous execution when disabled or when no loop is available.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    async def _run_sync(self, fn: Callable[[], Any]) -> Any:
        return fn()

    async def _gather(self, funcs: Iterable[Callable[[], Any]]) -> List[Any]:
        tasks = [asyncio.create_task(self._run_sync(fn)) for fn in funcs]
        return await asyncio.gather(*tasks)

    def run(self, funcs: Iterable[Callable[[], Any]]) -> List[Any]:
        if not self.enabled:
            return [fn() for fn in funcs]
        try:
            return asyncio.run(self._gather(funcs))
        except RuntimeError:
            # If an event loop is already running (e.g., inside a notebook), run synchronously.
            return [fn() for fn in funcs]
