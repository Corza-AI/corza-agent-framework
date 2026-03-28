"""
Corza Agent Framework — Context Health Monitor

Monitors context window utilization and triggers actions when thresholds
are crossed. Integrated into the engine's ReAct loop.

Health score = (token_ratio * 0.7) + (message_ratio * 0.3)
  - 0.0 = empty context
  - 1.0 = full context

Thresholds (configurable per agent):
  - 0.40: Start progressive compression
  - 0.80: Trigger full history compaction (LLM summarization)
  - 0.85: Warn agent to wrap up (one-shot injection)
  - 0.90: Hard stop — force one final response, no tools
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger("corza_agents.health")


@dataclass
class ContextHealthConfig:
    """
    Health thresholds for context window management.

    Attach to an AgentDefinition via metadata:
        agent = AgentDefinition(
            ...,
            metadata={"context_health": ContextHealthConfig(max_tokens=200_000)},
        )

    Or configure on the engine directly.
    """

    max_tokens: int = 200_000  # Model's context window
    max_messages: int = 100  # Soft cap for health score weighting

    # Threshold tiers (0.0 to 1.0)
    compress_threshold: float = 0.40  # Start progressive compression
    compact_threshold: float = 0.80  # Trigger LLM-based summarization
    warn_threshold: float = 0.85  # Warn agent to wrap up
    hard_stop_threshold: float = 0.90  # Force final turn, no tools


@dataclass
class HealthAction:
    """What the engine should do this turn based on context health."""

    should_compress: bool = False  # Apply progressive compression
    should_compact: bool = False  # Trigger LLM-based summarization
    should_warn_agent: bool = False  # Inject wrap-up warning
    should_hard_stop: bool = False  # Final turn, then break
    health_score: float = 0.0


def compute_health_score(
    total_input_tokens: int,
    message_count: int,
    config: ContextHealthConfig,
) -> float:
    """
    Compute context health score from 0.0 (empty) to 1.0 (full).

    Token usage is the primary signal (70% weight) because it directly
    measures what the API will reject. Message count (30% weight) catches
    cases where many small messages accumulate.
    """
    token_ratio = min(total_input_tokens / config.max_tokens, 1.0) if config.max_tokens > 0 else 0.0
    message_ratio = (
        min(message_count / config.max_messages, 1.0) if config.max_messages > 0 else 0.0
    )
    return (token_ratio * 0.7) + (message_ratio * 0.3)


def assess_health(
    total_input_tokens: int,
    message_count: int,
    config: ContextHealthConfig,
    already_warned: bool = False,
) -> HealthAction:
    """
    Determine what context management actions should happen this turn.

    Called every turn after the LLM response. Actions are NOT mutually
    exclusive — multiple can fire in one turn.

    Args:
        total_input_tokens: Cumulative input tokens so far
        message_count: Total messages in the session
        config: Health thresholds
        already_warned: Whether warn has already fired (one-shot)

    Returns:
        HealthAction with flags for each possible action
    """
    score = compute_health_score(total_input_tokens, message_count, config)
    action = HealthAction(health_score=score)

    if score >= config.compress_threshold:
        action.should_compress = True

    if score >= config.compact_threshold and message_count > 10:
        action.should_compact = True

    if score >= config.warn_threshold and not already_warned:
        action.should_warn_agent = True

    if score >= config.hard_stop_threshold:
        action.should_hard_stop = True

    if action.should_compress or action.should_warn_agent or action.should_hard_stop:
        log.info(
            "context_health_check",
            score=round(score, 3),
            tokens=total_input_tokens,
            messages=message_count,
            compress=action.should_compress,
            compact=action.should_compact,
            warn=action.should_warn_agent,
            hard_stop=action.should_hard_stop,
        )

    return action


# ══════════════════════════════════════════════════════════════════
# Warning / hard-stop messages injected into context
# ══════════════════════════════════════════════════════════════════

CONTEXT_WARNING_MESSAGE = (
    "[SYSTEM: Context window is running low. "
    "Wrap up your current line of investigation and provide a final summary. "
    "Avoid spawning new sub-agents or making large tool calls.]"
)

CONTEXT_HARD_STOP_MESSAGE = (
    "[SYSTEM: Context window is critically full. "
    "This is your FINAL turn. Provide your best summary with the information "
    "gathered so far. Do NOT call any tools.]"
)
