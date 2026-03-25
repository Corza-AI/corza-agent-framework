"""Tests for the AgentService layer and user/tenant scoping."""
import pytest

from corza_agents import AgentDefinition, AgentLLM, Orchestrator, ToolRegistry
from corza_agents.api.service import AgentService
from corza_agents.persistence.memory import InMemoryRepository


@pytest.fixture
async def service():
    repo = InMemoryRepository()
    await repo.initialize()
    llm = AgentLLM()
    tools = ToolRegistry()
    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    agents = {
        "assistant": AgentDefinition(name="assistant", model="ollama:test"),
    }
    return AgentService(orchestrator, agents)


@pytest.mark.asyncio
async def test_create_session(service):
    session = await service.create_session("assistant")
    assert session.id
    assert session.agent_id == "assistant"
    assert session.user_id == ""
    assert session.tenant_id == ""


@pytest.mark.asyncio
async def test_create_session_with_user(service):
    session = await service.create_session(
        "assistant", user_id="alice", tenant_id="acme"
    )
    assert session.user_id == "alice"
    assert session.tenant_id == "acme"


@pytest.mark.asyncio
async def test_create_session_with_metadata(service):
    session = await service.create_session(
        "assistant", metadata={"source": "web"}
    )
    assert session.metadata == {"source": "web"}
    assert session.user_id == ""  # metadata should NOT be passed as user_id


@pytest.mark.asyncio
async def test_create_session_unknown_agent(service):
    with pytest.raises(KeyError, match="not found"):
        await service.create_session("nonexistent")


@pytest.mark.asyncio
async def test_get_session(service):
    created = await service.create_session("assistant", user_id="alice")
    fetched = await service.get_session(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.user_id == "alice"


@pytest.mark.asyncio
async def test_get_sessions_for_user(service):
    await service.create_session("assistant", user_id="alice", tenant_id="acme")
    await service.create_session("assistant", user_id="alice", tenant_id="acme")
    await service.create_session("assistant", user_id="bob", tenant_id="acme")

    alice = await service.get_sessions_for_user("alice", "acme")
    assert len(alice) == 2

    bob = await service.get_sessions_for_user("bob", "acme")
    assert len(bob) == 1

    nobody = await service.get_sessions_for_user("nobody")
    assert len(nobody) == 0


@pytest.mark.asyncio
async def test_delete_session(service):
    session = await service.create_session("assistant")
    assert await service.get_session(session.id) is not None

    await service.delete_session(session.id)
    assert await service.get_session(session.id) is None


@pytest.mark.asyncio
async def test_list_agents(service):
    agents = service.list_agents()
    assert len(agents) == 1
    assert agents[0].name == "assistant"


@pytest.mark.asyncio
async def test_health(service):
    h = service.health()
    assert h["status"] == "ok"
    assert h["agents_count"] == 1
    assert isinstance(h["middleware"], list)
