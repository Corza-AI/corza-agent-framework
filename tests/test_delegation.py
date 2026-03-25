"""
Tests for orchestrator → sub-agent delegation and task agent restrictions.

Verifies that sub-agents (identified by parent_session_id) can only use
action='report', while the orchestrator can use all manage_agent actions.
"""
import pytest

from corza_agents.core.types import (
    AgentDefinition,
    ExecutionContext,
)
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response


async def _setup():
    """Create an Orchestrator with a registered sub-agent and return it."""
    llm = MockLLM([make_text_response("Ok.")])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    researcher = AgentDefinition(
        name="researcher",
        model="mock:test",
        description="Researches topics",
    )
    orchestrator.register_sub_agent("researcher", researcher)

    return orchestrator


def _get_manage_agent_handler(orchestrator: Orchestrator):
    """Extract the wired manage_agent handler from the tool registry."""
    tool_def = orchestrator._tools.get("manage_agent")
    assert tool_def is not None, "manage_agent tool not found in registry"
    assert tool_def.handler is not None, "manage_agent handler not wired"
    return tool_def.handler


def _sub_agent_context() -> ExecutionContext:
    """ExecutionContext that simulates a sub-agent (has parent_session_id)."""
    return ExecutionContext(
        session_id="child-456",
        agent_id="agent-abc",
        agent_name="researcher",
        parent_session_id="parent-123",
    )


def _orchestrator_context() -> ExecutionContext:
    """ExecutionContext that simulates the orchestrator (no parent_session_id)."""
    return ExecutionContext(
        session_id="orch-789",
        agent_id="agent-brain",
        agent_name="brain",
    )


# ══════════════════════════════════════════════════════════════════════
# Sub-agent restriction tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sub_agent_report_only_spawn_blocked():
    """Sub-agents cannot spawn other agents — only 'report' is allowed."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _sub_agent_context()

    result = await handler(action="spawn", agent_name="researcher", task="do stuff", ctx=ctx)

    assert result["status"] == "error"
    assert "report" in result["message"].lower()


@pytest.mark.asyncio
async def test_sub_agent_report_only_message_blocked():
    """Sub-agents cannot send messages to other agents."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _sub_agent_context()

    result = await handler(action="message", session_id="some-session", task="follow up", ctx=ctx)

    assert result["status"] == "error"
    assert "report" in result["message"].lower()


@pytest.mark.asyncio
async def test_sub_agent_report_only_list_blocked():
    """Sub-agents cannot list available agents."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _sub_agent_context()

    result = await handler(action="list", ctx=ctx)

    assert result["status"] == "error"
    assert "report" in result["message"].lower()


# ══════════════════════════════════════════════════════════════════════
# Orchestrator privilege tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_orchestrator_can_list():
    """The orchestrator (no parent_session_id) can list available agents."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _orchestrator_context()

    result = await handler(action="list", ctx=ctx)

    assert result["status"] == "success"
    assert "researcher" in result["agents"]


# ══════════════════════════════════════════════════════════════════════
# Sub-agent registration
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_sub_agent_registration():
    """register_sub_agent stores the definition and it appears in sub_agents."""
    llm = MockLLM([make_text_response("Ok.")])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)

    researcher = AgentDefinition(
        name="researcher",
        model="mock:test",
        description="Researches topics",
    )
    orchestrator.register_sub_agent("researcher", researcher)

    assert "researcher" in orchestrator.sub_agents
    assert orchestrator.sub_agents["researcher"].name == "researcher"


# ══════════════════════════════════════════════════════════════════════
# spawn_parallel tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_spawn_parallel_blocked_for_sub_agents():
    """Sub-agents cannot use spawn_parallel — only 'report' is allowed."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _sub_agent_context()

    result = await handler(action="spawn_parallel", tasks='[{"agent_name":"researcher","task":"do stuff"}]', ctx=ctx)

    assert result["status"] == "error"
    assert "report" in result["message"].lower()


@pytest.mark.asyncio
async def test_spawn_parallel_validation():
    """spawn_parallel with empty tasks returns an error."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _orchestrator_context()

    result = await handler(action="spawn_parallel", tasks="", ctx=ctx)

    assert result["status"] == "error"
    assert "tasks" in result["message"].lower() or "provide" in result["message"].lower()


@pytest.mark.asyncio
async def test_spawn_parallel_invalid_json():
    """spawn_parallel with non-JSON tasks returns an error."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _orchestrator_context()

    result = await handler(action="spawn_parallel", tasks="not-json", ctx=ctx)

    assert result["status"] == "error"
    assert "invalid" in result["message"].lower() or "json" in result["message"].lower()


@pytest.mark.asyncio
async def test_spawn_parallel_exceeds_limit():
    """spawn_parallel with more tasks than the limit returns an error."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _orchestrator_context()

    import json
    tasks = json.dumps([
        {"agent_name": "researcher", "task": f"task {i}"}
        for i in range(6)
    ])
    result = await handler(action="spawn_parallel", tasks=tasks, ctx=ctx)

    assert result["status"] == "error"
    assert "too many" in result["message"].lower() or "maximum" in result["message"].lower()


@pytest.mark.asyncio
async def test_spawn_parallel_unknown_agent():
    """spawn_parallel with a non-existent agent_name returns an error."""
    orchestrator = await _setup()
    handler = _get_manage_agent_handler(orchestrator)
    ctx = _orchestrator_context()

    import json
    tasks = json.dumps([
        {"agent_name": "nonexistent_agent", "task": "do something"}
    ])
    result = await handler(action="spawn_parallel", tasks=tasks, ctx=ctx)

    assert result["status"] == "error"
    assert "not found" in result["message"].lower()
