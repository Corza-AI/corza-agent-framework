"""
Tests for rate limit middleware.
"""
import pytest

from corza_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    MessageRole,
)
from corza_agents.middleware.rate_limit import RateLimitMiddleware


def _ctx(session_id: str = "s1", user_id: str = "u1", tenant_id: str = "t1"):
    return ExecutionContext(
        session_id=session_id,
        agent_id="test",
        agent_name="test",
        metadata={"user_id": user_id, "tenant_id": tenant_id},
    )


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    """Calls under the limit pass through."""
    mw = RateLimitMiddleware(max_calls=5, window_seconds=60)
    ctx = _ctx()
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    for _ in range(4):
        result_msgs, _ = await mw.before_llm_call(msgs, [], ctx)
        # Should pass through without rate limit message
        assert len(result_msgs) == 1


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    """Calls over the limit get rate limit injection."""
    mw = RateLimitMiddleware(max_calls=3, window_seconds=60)
    ctx = _ctx()
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    # Use up the limit
    for _ in range(3):
        await mw.before_llm_call(msgs, [], ctx)

    # 4th call should be rate limited
    result_msgs, _ = await mw.before_llm_call(msgs, [], ctx)
    assert len(result_msgs) == 2  # Original + rate limit message
    assert "Rate limit exceeded" in result_msgs[-1].content


@pytest.mark.asyncio
async def test_rate_limit_per_user_isolation():
    """Different users have independent rate limits."""
    mw = RateLimitMiddleware(max_calls=2, window_seconds=60, scope="user")
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    ctx1 = _ctx(user_id="alice")
    ctx2 = _ctx(user_id="bob")

    # Alice uses 2 calls
    await mw.before_llm_call(msgs, [], ctx1)
    await mw.before_llm_call(msgs, [], ctx1)

    # Alice is rate limited
    result, _ = await mw.before_llm_call(msgs, [], ctx1)
    assert "Rate limit" in result[-1].content

    # Bob is NOT rate limited
    result, _ = await mw.before_llm_call(msgs, [], ctx2)
    assert len(result) == 1  # No rate limit message


@pytest.mark.asyncio
async def test_rate_limit_tenant_scope():
    """Tenant scope groups all users under same tenant."""
    mw = RateLimitMiddleware(max_calls=2, window_seconds=60, scope="tenant")
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    ctx1 = _ctx(user_id="alice", tenant_id="acme")
    ctx2 = _ctx(user_id="bob", tenant_id="acme")

    await mw.before_llm_call(msgs, [], ctx1)
    await mw.before_llm_call(msgs, [], ctx2)

    # Both under same tenant — should be rate limited
    result, _ = await mw.before_llm_call(msgs, [], ctx1)
    assert "Rate limit" in result[-1].content


@pytest.mark.asyncio
async def test_rate_limit_usage_report():
    """get_usage() returns current state."""
    mw = RateLimitMiddleware(max_calls=10, window_seconds=60)
    ctx = _ctx()
    msgs = [AgentMessage(role=MessageRole.USER, content="Hi")]

    await mw.before_llm_call(msgs, [], ctx)
    await mw.before_llm_call(msgs, [], ctx)

    usage = mw.get_usage(ctx)
    assert usage["calls_used"] == 2
    assert usage["calls_remaining"] == 8
    assert usage["max_calls"] == 10


def test_rate_limit_name():
    mw = RateLimitMiddleware()
    assert mw.name == "RateLimitMiddleware"
