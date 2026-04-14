"""
Corza Agent Framework — Active Run Registry

Detaches agent execution from the HTTP/SSE connection that started it.

When a client POSTs a message, the orchestrator run is spawned as a top-level
asyncio Task owned by the registry, not by the request. Events are fanned out
to any number of subscriber queues. If a subscriber disconnects (client
navigates away, SSE cancelled), its queue is removed — but the producer task
keeps running until the agent loop completes naturally or is explicitly
cancelled via AgentService.cancel_run(). Events continue to be persisted to
the DB by the engine throughout.

This is the single fix that makes research / investigator / intelligence /
corza sessions survive page navigation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from corza_agents.streaming.events import StreamEvent

log = structlog.get_logger("corza_agents.api.run_registry")

_BACKLOG_MAX = 4096


@dataclass
class ActiveRun:
    """A live orchestrator run, decoupled from any particular HTTP request."""

    session_id: str
    producer: asyncio.Task[None] | None = None
    subscribers: list[asyncio.Queue[StreamEvent | None]] = field(default_factory=list)
    backlog: list[StreamEvent] = field(default_factory=list)
    done: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def subscribe(self) -> asyncio.Queue[StreamEvent | None]:
        # Unbounded queue: fan-out consumers (SSE endpoints) drain continuously,
        # and backpressure belongs at the HTTP / TCP layer, not here. A bounded
        # queue risks dropping events mid-stream and kicking subscribers out.
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        async with self.lock:
            for evt in self.backlog:
                queue.put_nowait(evt)
            if self.done:
                queue.put_nowait(None)
            else:
                self.subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[StreamEvent | None]) -> None:
        async with self.lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)

    async def publish(self, event: StreamEvent) -> None:
        async with self.lock:
            self.backlog.append(event)
            if len(self.backlog) > _BACKLOG_MAX:
                del self.backlog[: len(self.backlog) - _BACKLOG_MAX]
            for q in self.subscribers:
                q.put_nowait(event)

    async def close(self) -> None:
        async with self.lock:
            self.done = True
            for q in self.subscribers:
                q.put_nowait(None)
            self.subscribers.clear()


class RunRegistry:
    """Maps session_id -> ActiveRun. Module-level singleton per-process."""

    def __init__(self) -> None:
        self._runs: dict[str, ActiveRun] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> ActiveRun | None:
        async with self._lock:
            return self._runs.get(session_id)

    async def start(
        self,
        session_id: str,
        run_coro_factory: Any,
    ) -> tuple[ActiveRun, bool]:
        """
        Start a detached run for `session_id`. `run_coro_factory` must be a
        zero-arg callable returning an async iterator of StreamEvent.

        Returns `(active, is_new)`. If a run is already active for this
        session, returns it with `is_new=False` — the factory is NOT invoked
        and the caller's new content is NOT fed to the agent. Callers that
        need the new content to be processed must first wait for the existing
        run to finish (or cancel it) before calling `start` again.

        Create-and-register is atomic under `self._lock` to prevent a TOCTOU
        race where two concurrent starts both spawn producer tasks.
        """
        async with self._lock:
            existing = self._runs.get(session_id)
            if existing and not existing.done:
                return existing, False

            active = ActiveRun(session_id=session_id)

            async def _drive() -> None:
                try:
                    async for event in run_coro_factory():
                        await active.publish(event)
                except asyncio.CancelledError:
                    log.info("run_cancelled", session_id=session_id)
                    raise
                except Exception as e:  # pragma: no cover - defensive
                    log.error("run_failed", session_id=session_id, error=str(e)[:500])
                finally:
                    await active.close()
                    async with self._lock:
                        if self._runs.get(session_id) is active:
                            del self._runs[session_id]

            active.producer = asyncio.create_task(
                _drive(), name=f"agent-run:{session_id}"
            )
            self._runs[session_id] = active
            return active, True

    async def cancel(self, session_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(session_id)
        if not run or run.producer is None:
            return False
        run.producer.cancel()
        try:
            await run.producer
        except (asyncio.CancelledError, Exception):
            pass
        return True


_registry: RunRegistry | None = None


def get_registry() -> RunRegistry:
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry


async def subscribe_iter(active: ActiveRun) -> AsyncIterator[StreamEvent]:
    """Yield events for one subscriber. Auto-unsubscribes on cancel/exit."""
    queue = await active.subscribe()
    try:
        while True:
            event = await queue.get()
            if event is None:
                return
            yield event
    finally:
        await active.unsubscribe(queue)
