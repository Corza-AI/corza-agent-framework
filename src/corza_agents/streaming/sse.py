"""
Corza Agent Framework — Production SSE Streaming

Handles heartbeat, client disconnect detection, and event ID-based reconnection.
Used by the FastAPI router to stream agent events to the frontend.
"""
import asyncio
from collections.abc import AsyncIterator

from starlette.requests import Request
from starlette.responses import StreamingResponse

from corza_agents.streaming.events import StreamEvent

SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
}


async def sse_response(
    request: Request,
    events: AsyncIterator[StreamEvent],
    heartbeat_interval: int = 15,
) -> StreamingResponse:
    """
    Create a production SSE StreamingResponse with heartbeat and disconnect detection.

    Features:
    - Event IDs for reconnection (Last-Event-ID support)
    - Heartbeat comments every `heartbeat_interval` seconds
    - Client disconnect detection
    - Proper headers for SSE

    Usage in a FastAPI endpoint:
        @router.post("/sessions/{sid}/messages")
        async def send_message(sid: str, req: Request, ...):
            events = service.send_message(sid, content)
            return await sse_response(request, events)
    """
    last_event_id = request.headers.get("Last-Event-ID", "")

    async def _generate():
        skip_until_found = bool(last_event_id)

        try:
            async for event in events:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # Skip events already sent (reconnection support)
                if skip_until_found:
                    if event.id == last_event_id:
                        skip_until_found = False
                    continue

                yield event.to_sse()

                # Terminal events — close the stream
                if event.type.value in ("session.completed", "error"):
                    break

        except asyncio.CancelledError:
            pass

    async def _generate_with_heartbeat():
        """
        Interleave events with independent heartbeat comments.

        Uses an asyncio Queue so the heartbeat task and the event
        consumer run concurrently. This ensures heartbeats fire even
        during long LLM calls (60+ seconds) where no events flow.
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _heartbeat_pump():
            """Background task: push heartbeat comments at fixed intervals."""
            try:
                while True:
                    await asyncio.sleep(heartbeat_interval)
                    await queue.put(": heartbeat\n\n")
            except asyncio.CancelledError:
                pass

        async def _event_pump():
            """Consume agent events and push them to the queue."""
            try:
                async for chunk in _generate():
                    await queue.put(chunk)
            finally:
                await queue.put(None)  # Sentinel: stream is done

        heartbeat_task = asyncio.create_task(_heartbeat_pump())
        event_task = asyncio.create_task(_event_pump())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            heartbeat_task.cancel()
            event_task.cancel()
            # Suppress CancelledError from cleanup
            for t in (heartbeat_task, event_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        _generate_with_heartbeat(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
