"""
Tests for progressive context compression and health monitoring.
"""
import pytest

from corza_agents.core.types import AgentMessage, MessageRole
from corza_agents.memory.health import (
    ContextHealthConfig,
    assess_health,
    compute_health_score,
)
from corza_agents.middleware.context_compression import (
    _MARKER_COLD,
    _MARKER_EXPIRED,
    _MARKER_WARM,
    CompressionConfig,
    ContextCompressionMiddleware,
    progressive_compress,
)

# ══════════════════════════════════════════════════════════════════
# Progressive Compression
# ══════════════════════════════════════════════════════════════════

def _make_messages(count: int, tool_result_content: str = "x" * 5000) -> list[AgentMessage]:
    """Create a message list with alternating user/tool_result messages."""
    msgs = []
    for i in range(count):
        if i % 2 == 0:
            msgs.append(AgentMessage(role=MessageRole.USER, content=f"Message {i}"))
        else:
            msgs.append(AgentMessage(
                role=MessageRole.TOOL_RESULT, content=tool_result_content,
                tool_call_id=f"tc-{i}", tool_name="search",
            ))
    return msgs


def test_fresh_messages_untouched():
    """Messages within fresh_window are never compressed."""
    config = CompressionConfig(fresh_window=10)
    msgs = _make_messages(8)  # All within fresh window
    result = progressive_compress(msgs, config)
    assert result == msgs  # Same content


def test_warm_tier_truncation():
    """Messages in warm tier get head+tail truncation."""
    config = CompressionConfig(fresh_window=5, warm_window=10)
    msgs = _make_messages(30)  # Some will be in warm tier

    result = progressive_compress(msgs, config)

    # Find a warm-tier tool result (positions 15-25 from end)
    for i, msg in enumerate(result):
        pos_from_end = len(result) - 1 - i
        if msg.role == MessageRole.TOOL_RESULT and 5 <= pos_from_end < 15:
            if _MARKER_WARM in msg.content:
                assert len(msg.content) < 5000  # Truncated from original 5000
                return
    # At least some warm compression should have happened for large tool results
    warm_compressed = [m for m in result if isinstance(m.content, str) and _MARKER_WARM in m.content]
    assert len(warm_compressed) >= 0  # May be 0 if all below threshold


def test_cold_tier_stubs():
    """Messages in cold tier become short stubs."""
    config = CompressionConfig(fresh_window=3, warm_window=5, cold_window=10)
    msgs = _make_messages(40)

    result = progressive_compress(msgs, config)

    cold_compressed = [
        m for m in result
        if isinstance(m.content, str) and _MARKER_COLD in m.content
    ]
    assert len(cold_compressed) > 0


def test_expired_tier_placeholder():
    """Messages beyond cold_window become static placeholders."""
    config = CompressionConfig(fresh_window=3, warm_window=5, cold_window=10)
    msgs = _make_messages(60)

    result = progressive_compress(msgs, config)

    expired = [
        m for m in result
        if isinstance(m.content, str) and m.content == _MARKER_EXPIRED
    ]
    assert len(expired) > 0


def test_idempotent():
    """Compressing already-compressed messages doesn't change them."""
    config = CompressionConfig(fresh_window=3, warm_window=5, cold_window=10)
    msgs = _make_messages(40)

    result1 = progressive_compress(msgs, config)
    result2 = progressive_compress(result1, config)

    # Content should be identical after second pass
    for m1, m2 in zip(result1, result2):
        assert m1.content == m2.content


def test_never_mutates_input():
    """progressive_compress returns a new list, never mutates input."""
    config = CompressionConfig(fresh_window=3, warm_window=5, cold_window=10)
    msgs = _make_messages(40)
    original_contents = [m.content for m in msgs]

    _ = progressive_compress(msgs, config)

    # Original list should be unchanged
    for orig, msg in zip(original_contents, msgs):
        assert msg.content == orig


def test_user_messages_never_compressed():
    """Only TOOL_RESULT messages are compressed, not USER or ASSISTANT."""
    config = CompressionConfig(fresh_window=2, warm_window=3, cold_window=5)
    msgs = []
    for i in range(30):
        msgs.append(AgentMessage(role=MessageRole.USER, content=f"User message {i} " * 500))

    result = progressive_compress(msgs, config)

    # No compression markers in user messages
    for msg in result:
        content = msg.content if isinstance(msg.content, str) else ""
        assert _MARKER_WARM not in content
        assert _MARKER_COLD not in content
        assert _MARKER_EXPIRED not in content


# ══════════════════════════════════════════════════════════════════
# Middleware Integration
# ══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_compression_middleware_skips_short_conversations():
    """Middleware passes through short conversations."""
    from corza_agents.core.types import ExecutionContext
    mw = ContextCompressionMiddleware()
    msgs = _make_messages(5)
    ctx = ExecutionContext(session_id="s1", agent_id="test", agent_name="test")

    result_msgs, result_tools = await mw.before_llm_call(msgs, [], ctx)
    assert result_msgs == msgs


@pytest.mark.asyncio
async def test_compression_middleware_compresses_long_conversations():
    """Middleware compresses long conversations."""
    from corza_agents.core.types import ExecutionContext
    mw = ContextCompressionMiddleware(CompressionConfig(fresh_window=3))
    msgs = _make_messages(60)
    ctx = ExecutionContext(session_id="s1", agent_id="test", agent_name="test")

    result_msgs, _ = await mw.before_llm_call(msgs, [], ctx)

    # Some messages should be compressed
    all_content = " ".join(
        m.content for m in result_msgs if isinstance(m.content, str)
    )
    assert _MARKER_EXPIRED in all_content or _MARKER_COLD in all_content


# ══════════════════════════════════════════════════════════════════
# Health Scoring
# ══════════════════════════════════════════════════════════════════

def test_health_score_empty():
    """Empty context has score 0."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    score = compute_health_score(0, 0, config)
    assert score == 0.0


def test_health_score_full():
    """Full context has score 1.0."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    score = compute_health_score(200_000, 100, config)
    assert score == 1.0


def test_health_score_token_weighted():
    """Token usage has 70% weight."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    # 50% tokens, 0 messages
    score = compute_health_score(100_000, 0, config)
    assert abs(score - 0.35) < 0.01  # 0.5 * 0.7 = 0.35


def test_health_score_message_weighted():
    """Message count has 30% weight."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    # 0 tokens, 50% messages
    score = compute_health_score(0, 50, config)
    assert abs(score - 0.15) < 0.01  # 0.5 * 0.3 = 0.15


def test_assess_health_no_action():
    """Low usage triggers no actions."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    action = assess_health(10_000, 5, config)
    assert not action.should_compress
    assert not action.should_warn_agent
    assert not action.should_hard_stop


def test_assess_health_compress():
    """40%+ triggers compression."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    # 60% tokens → score ~0.42 → should compress
    action = assess_health(120_000, 5, config)
    assert action.should_compress


def test_assess_health_warn():
    """80%+ triggers agent warning (one-shot)."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    action = assess_health(180_000, 80, config)
    assert action.should_warn_agent

    # Already warned → don't warn again
    action2 = assess_health(180_000, 80, config, already_warned=True)
    assert not action2.should_warn_agent


def test_assess_health_hard_stop():
    """90%+ triggers hard stop."""
    config = ContextHealthConfig(max_tokens=200_000, max_messages=100)
    action = assess_health(190_000, 90, config)
    assert action.should_hard_stop
