"""
Tests for the ContextManager — context window management, truncation, and limits.
"""
import pytest

from corza_agents.core.types import (
    AgentMessage,
    MessageRole,
    ToolCall,
)
from corza_agents.memory.context import (
    ContextManager,
    estimate_message_tokens,
    estimate_tokens,
    get_context_limit,
)
from corza_agents.persistence.memory import InMemoryRepository


def test_estimate_tokens():
    """Token estimation: ~4 chars per token."""
    assert estimate_tokens("hello world") == 2  # 11 chars // 4
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100


def test_estimate_message_tokens_plain():
    """Estimate tokens for a plain text message."""
    msg = AgentMessage(role=MessageRole.USER, content="Hello world, this is a test.")
    tokens = estimate_message_tokens(msg)
    assert tokens > 0


def test_estimate_message_tokens_with_count():
    """If token_count is set, use that directly."""
    msg = AgentMessage(role=MessageRole.USER, content="Hello", token_count=42)
    assert estimate_message_tokens(msg) == 42


def test_estimate_message_tokens_with_tool_calls():
    """Tool calls add to token count."""
    msg = AgentMessage(
        role=MessageRole.ASSISTANT,
        content="Let me search.",
        tool_calls=[ToolCall(tool_name="search", arguments={"query": "big data"})],
    )
    tokens = estimate_message_tokens(msg)
    base_tokens = estimate_tokens("Let me search.")
    assert tokens > base_tokens  # tool calls add tokens


def test_get_context_limit_known_model():
    """Known models return their specific limit."""
    limit = get_context_limit("anthropic:claude-sonnet-4-6")
    assert limit == 200_000


def test_get_context_limit_unknown_model():
    """Unknown models return the default."""
    limit = get_context_limit("provider:unknown-model-xyz")
    assert limit == 128_000


def test_get_context_limit_custom_override():
    """Custom limits take priority."""
    custom = {"my-model": 50_000}
    limit = get_context_limit("provider:my-model", custom_limits=custom)
    assert limit == 50_000


@pytest.mark.asyncio
async def test_context_manager_under_limit():
    """Messages under the threshold pass through unchanged."""
    repo = InMemoryRepository()
    await repo.initialize()

    from corza_agents.core.types import AgentSession
    session = AgentSession(id="s1", agent_id="test")
    await repo.create_session(session)

    # Add a few short messages
    for i in range(3):
        msg = AgentMessage(session_id="s1", role=MessageRole.USER, content=f"Message {i}")
        await repo.add_message(msg)

    cm = ContextManager(repo)
    messages = await cm.build_context("s1", "System prompt", "openai:gpt-4.1")

    assert len(messages) == 3


@pytest.mark.asyncio
async def test_context_manager_truncates_old_tool_args():
    """Large tool arguments in old messages get truncated."""
    repo = InMemoryRepository()
    await repo.initialize()

    from corza_agents.core.types import AgentSession
    session = AgentSession(id="s1", agent_id="test")
    await repo.create_session(session)

    # Add an old message with huge tool arguments
    big_args = {"data": "x" * 10000}
    old_msg = AgentMessage(
        session_id="s1",
        role=MessageRole.ASSISTANT,
        content="Processing",
        tool_calls=[ToolCall(tool_name="big_tool", arguments=big_args)],
    )
    await repo.add_message(old_msg)

    # Add 6 recent messages to push the old one past the keep_recent threshold
    for i in range(6):
        await repo.add_message(
            AgentMessage(session_id="s1", role=MessageRole.USER, content=f"Recent {i}")
        )

    cm = ContextManager(repo, custom_context_limits={"tiny-model": 500})
    messages = await cm.build_context("s1", "System prompt", "provider:tiny-model")

    # The old tool call args should be truncated
    if messages[0].tool_calls:
        arg_val = messages[0].tool_calls[0].arguments.get("data", "")
        assert len(arg_val) < 10000


@pytest.mark.asyncio
async def test_session_purge():
    """Old sessions get purged based on age."""
    from datetime import UTC, datetime, timedelta

    repo = InMemoryRepository()
    await repo.initialize()

    from corza_agents.core.types import AgentSession, SessionStatus
    # Create an old completed session
    old_session = AgentSession(
        id="old-1", agent_id="test", status=SessionStatus.COMPLETED,
    )
    # Manually set old timestamp
    old_session = old_session.model_copy(update={
        "updated_at": datetime.now(UTC) - timedelta(hours=200)
    })
    repo._sessions["old-1"] = old_session

    # Create a recent session
    recent = AgentSession(id="recent-1", agent_id="test")
    await repo.create_session(recent)

    # Purge sessions older than 168 hours (1 week)
    count = await repo.purge_old_sessions(max_age_hours=168)
    assert count == 1

    assert await repo.get_session("old-1") is None
    assert await repo.get_session("recent-1") is not None
