"""
Corza Agent Framework — Progressive Context Compression Middleware

Applies tiered aging to tool results based on message position within
the conversation. Runs before each LLM call to keep context within
the model's window.

4-tier compression strategy:
  - Fresh  (last 10 messages): Full tool results preserved
  - Warm   (11-25 back):       Head (2KB) + tail (500 chars) truncation
  - Cold   (26-50 back):       150-char summary stubs
  - Expired (50+ back):        Static placeholder

Design invariants:
  - Never removes messages — only compresses content strings
  - Idempotent — markers detect already-compressed results
  - Immutable — returns a new list, never mutates input
  - No LLM calls — deterministic string operations only
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from corza_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    MessageRole,
    ToolSchema,
)
from corza_agents.middleware.base import BaseMiddleware

log = structlog.get_logger("corza_agents.context_compression")


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════


@dataclass
class CompressionConfig:
    """
    Per-agent compression thresholds. Tune based on your model's
    context window and typical conversation length.
    """

    # Tier boundaries (measured from END of message list)
    fresh_window: int = 10  # Last N messages: never touched
    warm_window: int = 25  # 11-25 back: head+tail truncation
    cold_window: int = 50  # 26-50 back: stub summaries
    # Beyond cold_window → expired (static placeholder)

    # Warm tier: how much to keep
    warm_head_chars: int = 2000
    warm_tail_chars: int = 500
    warm_skip_below: int = 2500  # Don't truncate results shorter than this

    # Cold tier: stub length
    cold_summary_chars: int = 150


# Default config — works well for 128K-200K context models
DEFAULT_COMPRESSION_CONFIG = CompressionConfig()

# Markers for idempotent detection
_MARKER_EXPIRED = "[Result compressed — no longer in context]"
_MARKER_COLD = "[Result compressed"
_MARKER_WARM = "…(truncated)"


# ══════════════════════════════════════════════════════════════════
# Middleware
# ══════════════════════════════════════════════════════════════════


class ContextCompressionMiddleware(BaseMiddleware):
    """
    Applies progressive compression to tool results before each LLM call.

    Usage:
        from corza_agents.middleware.context_compression import (
            ContextCompressionMiddleware, CompressionConfig,
        )

        # Default thresholds (10/25/50 message tiers)
        mw = ContextCompressionMiddleware()

        # Custom thresholds for smaller models
        mw = ContextCompressionMiddleware(CompressionConfig(
            fresh_window=5, warm_window=15, cold_window=30,
        ))

        orchestrator = Orchestrator(llm, tools, repo, middleware=[mw])
    """

    def __init__(self, config: CompressionConfig | None = None):
        self._config = config or DEFAULT_COMPRESSION_CONFIG

    @property
    def name(self) -> str:
        return "ContextCompressionMiddleware"

    async def before_llm_call(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        context: ExecutionContext,
    ) -> tuple[list[AgentMessage], list[ToolSchema]]:
        """Apply progressive compression to old tool results."""
        if not messages or len(messages) <= self._config.fresh_window:
            return messages, tools

        compressed = progressive_compress(messages, self._config)
        return compressed, tools


# ══════════════════════════════════════════════════════════════════
# Compression Logic
# ══════════════════════════════════════════════════════════════════


def progressive_compress(
    messages: list[AgentMessage],
    config: CompressionConfig | None = None,
) -> list[AgentMessage]:
    """
    Apply tiered aging to tool results based on message position.

    Returns a NEW list — never mutates input. Idempotent via markers.

    Tiers (measured from END of message list):
      - Fresh  (last ``fresh_window``): unchanged
      - Warm   (next ``warm_window``): head + tail truncation
      - Cold   (next ``cold_window``): 1-line stub
      - Expired (beyond ``cold_window``): static placeholder
    """
    if not config:
        config = DEFAULT_COMPRESSION_CONFIG
    if not messages:
        return messages

    total = len(messages)
    result = list(messages)  # Shallow copy — only replace changed items

    for i, msg in enumerate(result):
        # Only compress tool results
        if msg.role != MessageRole.TOOL_RESULT:
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Calculate position from end (0 = most recent)
        pos_from_end = total - 1 - i

        # Fresh tier — sacred, never touch
        if pos_from_end < config.fresh_window:
            continue

        # Warm tier — head + tail truncation
        if pos_from_end < config.fresh_window + config.warm_window:
            compressed = _compress_warm(content, config)
            if compressed is not content:
                result[i] = msg.model_copy(update={"content": compressed, "token_count": 0})

        # Cold tier — stub summary
        elif pos_from_end < config.fresh_window + config.cold_window:
            compressed = _compress_cold(content, config)
            if compressed is not content:
                result[i] = msg.model_copy(update={"content": compressed, "token_count": 0})

        # Expired tier — static placeholder
        else:
            if _MARKER_EXPIRED not in content:
                result[i] = msg.model_copy(update={"content": _MARKER_EXPIRED, "token_count": 0})

    return result


def _compress_warm(text: str, config: CompressionConfig) -> str:
    """Warm tier: head + tail truncation. Returns identity if no change needed."""
    # Already compressed at any tier → skip
    if _MARKER_COLD in text or _MARKER_EXPIRED in text or _MARKER_WARM in text:
        return text
    # Small results: leave alone
    if len(text) <= config.warm_skip_below:
        return text
    head = text[: config.warm_head_chars]
    tail = text[-config.warm_tail_chars :]
    omitted = len(text) - config.warm_head_chars - config.warm_tail_chars
    return f"{head}\n\n[... {omitted} chars omitted {_MARKER_WARM}]\n\n{tail}"


def _compress_cold(text: str, config: CompressionConfig) -> str:
    """Cold tier: 1-line summary stub. Returns identity if no change needed."""
    # Already cold or expired → skip
    if _MARKER_COLD in text or _MARKER_EXPIRED in text:
        return text
    first_line = text.split("\n", 1)[0][: config.cold_summary_chars].strip()
    if not first_line:
        return "[Result compressed: (empty)]"
    total_chars = len(text)
    return f"[Result compressed: {first_line} — {total_chars} chars total]"
