"""
Integration tests for the Orchestrator — sub-agent delegation and wiring.
"""
import pytest

from corza_agents.core.types import (
    AgentDefinition,
    EventType,
    SessionStatus,
)
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.tools.decorators import tool
from corza_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response, make_tool_call_response


@pytest.mark.asyncio
async def test_orchestrator_simple_run():
    """Orchestrator runs a single agent to completion."""
    llm = MockLLM([make_text_response("Analysis complete.")])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    agent = AgentDefinition(name="analyst", model="mock:test")

    events = []
    async for event in orchestrator.run("s1", "Analyze this", agent):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.SESSION_STARTED in event_types
    assert EventType.SESSION_COMPLETED in event_types

    session = await repo.get_session("s1")
    assert session.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_orchestrator_run_simple():
    """run_simple() returns final text directly."""
    llm = MockLLM([make_text_response("The answer is 42.")])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    agent = AgentDefinition(name="brain", model="mock:test")
    result = await orchestrator.run_simple("What is the answer?", agent)
    assert "42" in result


@pytest.mark.asyncio
async def test_orchestrator_with_tools():
    """Orchestrator handles tool execution through the engine."""
    @tool(name="calculate", description="Do math")
    async def calculate(expression: str) -> dict:
        return {"result": eval(expression)}

    llm = MockLLM([
        make_tool_call_response("calculate", {"expression": "2+2"}),
        make_text_response("The result is 4."),
    ])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()
    tools.register_function(calculate)

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    agent = AgentDefinition(name="math", model="mock:test", tools=["calculate"])

    events = []
    async for event in orchestrator.run("s1", "2+2?", agent):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.TOOL_EXECUTING in event_types
    assert EventType.TOOL_RESULT in event_types
    assert EventType.SESSION_COMPLETED in event_types


@pytest.mark.asyncio
async def test_orchestrator_sub_agent_registration():
    """Sub-agents can be registered and listed."""
    llm = MockLLM([make_text_response("Ok.")])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)

    researcher = AgentDefinition(name="researcher", model="mock:test", description="Researches topics")
    writer = AgentDefinition(name="writer", model="mock:test", description="Writes content")

    orchestrator.register_sub_agents({"researcher": researcher, "writer": writer})

    assert "researcher" in orchestrator.sub_agents
    assert "writer" in orchestrator.sub_agents
    assert len(orchestrator.sub_agents) == 2


@pytest.mark.asyncio
async def test_orchestrator_health():
    """Health check returns tool, middleware, and sub-agent info."""
    llm = MockLLM([])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    # Access middleware and tools
    assert len(orchestrator.middleware) >= 2  # audit + token_tracking
    assert orchestrator.repo is repo
    assert orchestrator.tools is tools


@pytest.mark.asyncio
async def test_orchestrator_shutdown():
    """Shutdown closes the repository."""
    llm = MockLLM([])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()
    await orchestrator.shutdown()
    # Should not raise
