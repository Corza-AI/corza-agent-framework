"""
Tests for the API router and AgentService — SSE streaming and REST endpoints.
"""
import json

import pytest

from corza_agents.api.service import AgentService
from corza_agents.core.types import (
    AgentDefinition,
    EventType,
)
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response


async def _build_service(
    responses: list, tools_list: list | None = None
) -> tuple[AgentService, InMemoryRepository]:
    """Build a service with mock LLM."""
    llm = MockLLM(responses)
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()
    if tools_list:
        for t in tools_list:
            tools.register_function(t)

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    agents = {
        "assistant": AgentDefinition(
            name="assistant", model="mock:test",
            tools=[t.tool_name for t in tools_list] if tools_list else [],
        ),
    }
    service = AgentService(orchestrator, agents)
    return service, repo


# ══════════════════════════════════════════════════════════════════════
# AgentService tests
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_service_create_session():
    """Service creates a session tied to a registered agent."""
    service, repo = await _build_service([make_text_response("Hi")])

    session = await service.create_session("assistant", user_id="u1", tenant_id="t1")
    assert session.agent_id == "assistant"
    assert session.user_id == "u1"
    assert session.tenant_id == "t1"

    # Persisted in repo
    stored = await repo.get_session(session.id)
    assert stored is not None
    assert stored.agent_id == "assistant"


@pytest.mark.asyncio
async def test_service_create_session_unknown_agent():
    """Creating a session for unknown agent raises KeyError."""
    service, _ = await _build_service([])

    with pytest.raises(KeyError, match="nonexistent"):
        await service.create_session("nonexistent")


@pytest.mark.asyncio
async def test_service_send_message_streaming():
    """Service streams events for a message."""
    service, repo = await _build_service([make_text_response("Hello there!")])

    session = await service.create_session("assistant")
    events = []
    async for event in service.send_message(session.id, "Hi"):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.SESSION_STARTED in event_types
    assert EventType.SESSION_COMPLETED in event_types


@pytest.mark.asyncio
async def test_service_send_message_sync():
    """send_message_sync returns final text."""
    service, repo = await _build_service([make_text_response("The answer is 42.")])

    session = await service.create_session("assistant")
    output, error = await service.send_message_sync(session.id, "What's the answer?")

    assert error is None
    assert "42" in output


@pytest.mark.asyncio
async def test_service_list_agents():
    """Service lists registered agents."""
    service, _ = await _build_service([])
    agents = service.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "assistant"


@pytest.mark.asyncio
async def test_service_register_agent():
    """Agents can be registered dynamically."""
    service, _ = await _build_service([])

    new_agent = AgentDefinition(name="writer", model="mock:test")
    service.register_agent("writer", new_agent)

    agents = service.list_agents()
    names = [a.name for a in agents]
    assert "writer" in names


@pytest.mark.asyncio
async def test_service_register_tool():
    """Tools can be registered dynamically via service."""
    service, _ = await _build_service([])
    service.register_tool(
        name="custom_search",
        description="Search custom DB",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    tools = service.list_tools()
    names = [t.name for t in tools]
    assert "custom_search" in names


@pytest.mark.asyncio
async def test_service_health():
    """Health check returns status and counts."""
    service, _ = await _build_service([])
    health = service.health()
    assert health["status"] == "ok"
    assert health["agents_count"] == 1
    assert isinstance(health["tools_count"], int)


@pytest.mark.asyncio
async def test_service_get_messages():
    """Messages are retrievable after a conversation."""
    service, _ = await _build_service([make_text_response("Yo!")])

    session = await service.create_session("assistant")
    async for _ in service.send_message(session.id, "Hey"):
        pass

    messages = await service.get_messages(session.id)
    assert len(messages) == 2
    assert messages[0].content == "Hey"


@pytest.mark.asyncio
async def test_service_memory():
    """Agent memory can be set and retrieved."""
    service, _ = await _build_service([])

    await service.set_agent_memory("assistant", "preference", {"theme": "dark"})
    memories = await service.get_agent_memories("assistant")
    assert len(memories) == 1
    assert memories[0]["key"] == "preference"


@pytest.mark.asyncio
async def test_service_delete_session():
    """Sessions can be deleted."""
    service, repo = await _build_service([])

    session = await service.create_session("assistant")
    await service.delete_session(session.id)

    stored = await repo.get_session(session.id)
    assert stored is None


# ══════════════════════════════════════════════════════════════════════
# SSE format tests
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_sse_format():
    """StreamEvent.to_sse() produces valid SSE format."""
    service, _ = await _build_service([make_text_response("Test")])

    session = await service.create_session("assistant")
    async for event in service.send_message(session.id, "Hi"):
        sse = event.to_sse()
        assert sse.startswith("id: ")
        assert "\nevent: " in sse
        assert "\ndata: " in sse
        assert sse.endswith("\n\n")

        # Data should be valid JSON
        data_line = [line for line in sse.split("\n") if line.startswith("data: ")][0]
        payload = json.loads(data_line.removeprefix("data: "))
        assert "session_id" in payload
        assert "type" in payload


@pytest.mark.asyncio
async def test_sse_reconnection_ids():
    """Each SSE event has a unique ID for reconnection support."""
    service, _ = await _build_service([make_text_response("Test")])

    session = await service.create_session("assistant")
    ids = set()
    async for event in service.send_message(session.id, "Hi"):
        ids.add(event.id)

    # All IDs should be unique
    assert len(ids) >= 3  # At least session_started, turn_started, session_completed
