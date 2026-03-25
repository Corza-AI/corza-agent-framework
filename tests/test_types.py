"""Tests for core types."""
from corza_agents import (
    AgentDefinition,
    AgentMessage,
    AgentSession,
    ExecutionContext,
    LLMUsage,
    MessageRole,
    Skill,
    ToolCall,
    ToolResult,
    ToolStatus,
)


def test_agent_definition_no_default_model():
    agent = AgentDefinition(name="test")
    assert agent.model == ""


def test_agent_definition_explicit_model():
    agent = AgentDefinition(name="test", model="openai:gpt-4o")
    assert agent.model == "openai:gpt-4o"


def test_agent_definition_max_llm_retries_default():
    agent = AgentDefinition(name="test")
    assert agent.max_llm_retries == 3


def test_agent_definition_callable_knowledge():
    agent = AgentDefinition(
        name="test",
        knowledge=["file.md", {"title": "Inline", "content": "data"}, lambda: "dynamic"],
    )
    assert len(agent.knowledge) == 3


def test_agent_message_text():
    msg = AgentMessage(role=MessageRole.USER, content="Hello")
    assert msg.text() == "Hello"


def test_agent_message_text_from_blocks():
    msg = AgentMessage(
        role=MessageRole.ASSISTANT,
        content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}],
    )
    assert msg.text() == "Hello\n world"


def test_tool_call_defaults():
    tc = ToolCall(tool_name="search", arguments={"q": "test"})
    assert tc.tool_name == "search"
    assert tc.id  # UUID generated


def test_tool_result_defaults():
    tr = ToolResult(tool_call_id="abc", tool_name="search", output="found it")
    assert tr.status == ToolStatus.SUCCESS


def test_llm_usage_defaults():
    usage = LLMUsage()
    assert usage.total_tokens == 0


def test_execution_context_skills():
    ctx = ExecutionContext(session_id="test")
    assert ctx.active_skills == []

    skill = Skill(name="test_skill", prompt_template="Do the thing")
    ctx.add_skill(skill)
    assert len(ctx.active_skills) == 1

    ctx.remove_skill("test_skill")
    assert len(ctx.active_skills) == 0


def test_session_defaults():
    session = AgentSession()
    assert session.total_cost_usd == 0.0
    assert session.turn_count == 0
    assert session.user_id == ""
    assert session.tenant_id == ""


def test_session_with_user():
    session = AgentSession(user_id="alice", tenant_id="acme")
    assert session.user_id == "alice"
    assert session.tenant_id == "acme"


def test_user_context():
    from corza_agents import UserContext
    ctx = UserContext(user_id="u1", tenant_id="t1", org_id="o1")
    assert ctx.user_id == "u1"
    assert ctx.tenant_id == "t1"
    assert ctx.org_id == "o1"
