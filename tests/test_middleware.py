"""
Tests for middleware — audit, permissions, token tracking.
"""
import pytest

from corza_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolStatus,
)
from corza_agents.middleware.audit import AuditMiddleware
from corza_agents.middleware.base import BaseMiddleware
from corza_agents.middleware.permissions import PermissionMiddleware, PermissionRule
from corza_agents.middleware.token_tracking import TokenTrackingMiddleware
from corza_agents.persistence.memory import InMemoryRepository


def _make_context(session_id: str = "s1") -> ExecutionContext:
    return ExecutionContext(
        session_id=session_id,
        agent_id="test-agent",
        agent_name="test",
        turn_number=1,
    )


# ══════════════════════════════════════════════════════════════════
# BaseMiddleware — pass-through behavior
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_base_middleware_passthrough():
    """BaseMiddleware returns inputs unchanged."""
    mw = BaseMiddleware()

    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]
    tools = [ToolSchema(name="t1", description="test")]
    ctx = _make_context()

    result_msgs, result_tools = await mw.before_llm_call(msgs, tools, ctx)
    assert result_msgs == msgs
    assert result_tools == tools


@pytest.mark.asyncio
async def test_base_middleware_name():
    """BaseMiddleware name defaults to class name."""
    assert BaseMiddleware().name == "BaseMiddleware"


# ══════════════════════════════════════════════════════════════════
# PermissionMiddleware
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_permission_middleware_allow():
    """Allowed tools pass through."""
    rules = [PermissionRule(tool_pattern="search_*", level="auto_approve")]
    mw = PermissionMiddleware(rules)
    ctx = _make_context()

    tc = ToolCall(tool_name="search_db", arguments={"q": "test"})
    result = await mw.before_tool_call(tc, ctx)
    assert result is not None
    assert result.tool_name == "search_db"


@pytest.mark.asyncio
async def test_permission_middleware_deny():
    """Denied tools return None (blocked)."""
    rules = [PermissionRule(tool_pattern="dangerous_*", level="deny")]
    mw = PermissionMiddleware(rules)
    ctx = _make_context()

    tc = ToolCall(tool_name="dangerous_delete", arguments={})
    result = await mw.before_tool_call(tc, ctx)
    assert result is None


@pytest.mark.asyncio
async def test_permission_middleware_no_matching_rule():
    """Tools without matching rules use default level."""
    rules = [PermissionRule(tool_pattern="admin_*", level="deny")]
    mw = PermissionMiddleware(rules, default_level="auto_approve")
    ctx = _make_context()

    tc = ToolCall(tool_name="search_db", arguments={})
    result = await mw.before_tool_call(tc, ctx)
    assert result is not None  # default allows


# ══════════════════════════════════════════════════════════════════
# AuditMiddleware
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_audit_middleware_logs_llm_call():
    """Audit middleware logs LLM calls to the repository."""
    repo = InMemoryRepository()
    await repo.initialize()

    from corza_agents.core.types import AgentSession
    await repo.create_session(AgentSession(id="s1", agent_id="test"))

    mw = AuditMiddleware(repo)
    ctx = _make_context()

    response = LLMResponse(
        content="Hello",
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        model="openai:gpt-4.1",
    )

    result = await mw.after_llm_call(response, ctx)
    assert result == response  # pass-through

    audit = await repo.get_audit_log("s1")
    assert len(audit) >= 1
    assert audit[0]["event_type"] == "llm_call"


@pytest.mark.asyncio
async def test_audit_middleware_logs_tool_call():
    """Audit middleware logs tool executions."""
    repo = InMemoryRepository()
    await repo.initialize()
    await repo.create_session(
        __import__("corza_agents.core.types", fromlist=["AgentSession"]).AgentSession(
            id="s1", agent_id="test"
        )
    )

    mw = AuditMiddleware(repo)
    ctx = _make_context()

    tc = ToolCall(tool_name="search", arguments={"q": "test"})
    tr = ToolResult(
        tool_call_id=tc.id, tool_name="search",
        output={"results": []}, status=ToolStatus.SUCCESS, duration_ms=50.0,
    )

    result = await mw.after_tool_call(tc, tr, ctx)
    assert result == tr

    audit = await repo.get_audit_log("s1")
    tool_audits = [a for a in audit if a["event_type"] == "tool_execution"]
    assert len(tool_audits) >= 1


# ══════════════════════════════════════════════════════════════════
# TokenTrackingMiddleware
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_token_tracking_middleware():
    """Token tracking updates session totals."""
    repo = InMemoryRepository()
    await repo.initialize()
    await repo.create_session(
        __import__("corza_agents.core.types", fromlist=["AgentSession"]).AgentSession(
            id="s1", agent_id="test"
        )
    )

    mw = TokenTrackingMiddleware(repo)
    ctx = _make_context()

    response = LLMResponse(
        content="Hello",
        usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        model="openai:gpt-4.1",
    )

    await mw.after_llm_call(response, ctx)

    session = await repo.get_session("s1")
    assert session.total_input_tokens == 100
    assert session.total_output_tokens == 50


# ══════════════════════════════════════════════════════════════════
# Custom Middleware
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_custom_middleware():
    """Users can subclass BaseMiddleware and override specific hooks."""
    injected = {}

    class InjectContextMiddleware(BaseMiddleware):
        @property
        def name(self) -> str:
            return "InjectContext"

        async def before_llm_call(self, messages, tools, context):
            injected["called"] = True
            injected["turn"] = context.turn_number
            return messages, tools

    mw = InjectContextMiddleware()
    ctx = _make_context()
    ctx.turn_number = 5

    msgs = [AgentMessage(role=MessageRole.USER, content="test")]
    await mw.before_llm_call(msgs, [], ctx)

    assert injected["called"] is True
    assert injected["turn"] == 5
