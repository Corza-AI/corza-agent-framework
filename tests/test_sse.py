"""
Tests for the SSE streaming module.

Verifies heartbeat, disconnect detection, and event formatting.
"""
from collections.abc import AsyncIterator

import pytest

from corza_agents.core.types import EventType
from corza_agents.streaming.events import StreamEvent
from corza_agents.streaming.sse import sse_response


class FakeRequest:
    """
    Minimal Request stub for SSE tests.

    Simulates a client that disconnects after `disconnect_after` calls
    to is_disconnected().
    """

    def __init__(self, disconnect_after: int = 999):
        self._call_count = 0
        self._disconnect_after = disconnect_after
        self.headers = {}

    async def is_disconnected(self) -> bool:
        self._call_count += 1
        return self._call_count > self._disconnect_after


def _make_event(session_id: str = "s1", event_type: EventType = EventType.TURN_STARTED) -> StreamEvent:
    return StreamEvent(type=event_type, session_id=session_id, data={})


async def _iter_events(events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_sse_client_disconnect_stops_stream():
    """Stream stops after the client disconnects (is_disconnected returns True)."""
    # Client disconnects after the first event is checked
    request = FakeRequest(disconnect_after=1)

    events = [
        _make_event(event_type=EventType.TURN_STARTED),
        _make_event(event_type=EventType.TURN_COMPLETED),
        _make_event(event_type=EventType.SESSION_COMPLETED),
    ]

    response = await sse_response(request, _iter_events(events))

    # Collect all chunks from the body_iterator
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    # Only 1 event should have been yielded before disconnect was detected
    # (heartbeat chunks are comments starting with ":", SSE events contain "event:")
    sse_chunks = [c for c in chunks if "event:" in c]
    assert len(sse_chunks) == 1
