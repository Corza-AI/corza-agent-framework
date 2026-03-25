"""
Tests for error paths — LLM timeout, retries, tool failures, session locking.
"""
import asyncio
from collections.abc import AsyncIterator

import pytest

from corza_agents.core.engine import AgentEngine
from corza_agents.core.errors import LLMError
from corza_agents.core.types import (
    AgentDefinition,
    EventType,
    LLMStreamChunk,
    SessionStatus,
)
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.skills.manager import SkillsManager
from corza_agents.tools.decorators import tool
from corza_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response

# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

async def _build_engine(mock_llm, tools=None):
    repo = InMemoryRepository()
    await repo.initialize()
    tool_registry = ToolRegistry()
    if tools:
        for t in tools:
            tool_registry.register_function(t)
    engine = AgentEngine(
        llm=mock_llm, tool_registry=tool_registry, repository=repo,
        skills_manager=SkillsManager(), middleware=[],
    )
    return engine, repo


class HangingLLM:
    """LLM that hangs forever (for timeout tests)."""
    async def stream_with_tools(self, **kwargs) -> AsyncIterator[LLMStreamChunk]:
        await asyncio.sleep(9999)
        yield LLMStreamChunk(type="text_delta", text="never")  # pragma: no cover

    async def count_tokens(self, text: str, model: str = "") -> int:
        return 10


class FailOnceLLM:
    """LLM that fails on first call, succeeds on retry."""
    def __init__(self):
        self._call_count = 0

    async def stream_with_tools(self, **kwargs) -> AsyncIterator[LLMStreamChunk]:
        self._call_count += 1
        if self._call_count == 1:
            raise LLMError("Transient failure", retryable=True)
        for chunk in make_text_response("Recovered successfully."):
            yield chunk

    async def count_tokens(self, text: str, model: str = "") -> int:
        return 10


# ══════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_llm_timeout():
    """LLM calls that exceed timeout are caught gracefully."""
    engine, repo = await _build_engine(HangingLLM())
    agent = AgentDefinition(name="test", model="mock:test", llm_timeout_seconds=0.1)

    events = []
    async for event in engine.run("s1", "Hi", agent):
        events.append(event)

    # Should get an error event, not hang forever
    event_types = [e.type for e in events]
    assert EventType.ERROR in event_types or EventType.SESSION_COMPLETED in event_types

    session = await repo.get_session("s1")
    assert session is not None
    # Session should be in WAITING_INPUT (recoverable) or FAILED
    assert session.status in (SessionStatus.WAITING_INPUT, SessionStatus.FAILED,
                               SessionStatus.COMPLETED)


@pytest.mark.asyncio
async def test_llm_retry_on_transient_error():
    """LLM retries on transient errors and eventually succeeds."""
    llm = FailOnceLLM()
    engine, repo = await _build_engine(llm)
    agent = AgentDefinition(name="test", model="mock:test", max_llm_retries=2)

    events = []
    async for event in engine.run("s1", "Hi", agent):
        events.append(event)

    # Should succeed after retry
    event_types = [e.type for e in events]
    assert EventType.SESSION_COMPLETED in event_types

    session = await repo.get_session("s1")
    assert session.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_session_locking_prevents_concurrent_writes():
    """Two concurrent runs on the same session are serialized (not interleaved)."""
    call_order = []

    class SlowLLM:
        def __init__(self):
            self._call_count = 0

        async def stream_with_tools(self, **kwargs) -> AsyncIterator[LLMStreamChunk]:
            self._call_count += 1
            call_id = self._call_count
            call_order.append(f"start-{call_id}")
            await asyncio.sleep(0.05)  # Simulate LLM latency
            call_order.append(f"end-{call_id}")
            for chunk in make_text_response(f"Response {call_id}"):
                yield chunk

        async def count_tokens(self, text: str, model: str = "") -> int:
            return 10

    engine, repo = await _build_engine(SlowLLM())
    agent = AgentDefinition(name="test", model="mock:test")

    # Run two messages concurrently on the same session
    async def run1():
        return [e async for e in engine.run("s1", "First", agent)]

    async def run2():
        return [e async for e in engine.run("s1", "Second", agent)]

    await asyncio.gather(run1(), run2())

    # The lock should serialize them: start-1, end-1, start-2, end-2
    # (not interleaved like start-1, start-2, end-1, end-2)
    assert call_order[0] == "start-1" or call_order[0] == "start-2"
    # First call should complete before second starts
    first_end_idx = next(i for i, x in enumerate(call_order) if x.startswith("end-"))
    second_start_idx = next(
        (i for i, x in enumerate(call_order) if x.startswith("start-") and i > 0), None
    )
    if second_start_idx is not None:
        assert first_end_idx < second_start_idx, \
            f"Sessions interleaved: {call_order}"


@pytest.mark.asyncio
async def test_tool_timeout_is_handled():
    """Tools that exceed their timeout get a TIMEOUT result."""
    @tool(name="slow_tool", description="Takes forever", timeout_seconds=1)
    async def slow_tool(x: str) -> dict:
        await asyncio.sleep(9999)
        return {"done": True}  # pragma: no cover

    from tests.helpers import make_tool_call_response
    llm = MockLLM([
        make_tool_call_response("slow_tool", {"x": "test"}),
        make_text_response("Tool timed out, but I'm fine."),
    ])
    engine, repo = await _build_engine(llm, tools=[slow_tool])
    agent = AgentDefinition(name="test", model="mock:test", tools=["slow_tool"])

    events = []
    async for event in engine.run("s1", "Go", agent):
        events.append(event)

    # Should have a tool result with timeout status
    tool_results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert len(tool_results) == 1
    assert tool_results[0].data["status"] in ("timeout", "error")


@pytest.mark.asyncio
async def test_llm_timeout_field_on_agent_definition():
    """llm_timeout_seconds is configurable on AgentDefinition."""
    agent = AgentDefinition(name="test", model="mock:test", llm_timeout_seconds=30.0)
    assert agent.llm_timeout_seconds == 30.0

    agent2 = AgentDefinition(name="test2", model="mock:test")
    assert agent2.llm_timeout_seconds == 120.0  # default
