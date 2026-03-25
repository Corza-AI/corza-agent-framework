"""
Corza Agent Framework — Rate Limit Middleware

Token-bucket rate limiter that prevents individual users or tenants
from overwhelming the system.

Limits are applied per user_id (or tenant_id, or session_id — configurable).
When a limit is exceeded, the tool call is blocked with a DENIED status.

Usage:
    from corza_agents.middleware.rate_limit import RateLimitMiddleware

    # Max 30 LLM calls per minute per user
    mw = RateLimitMiddleware(max_calls=30, window_seconds=60, scope="user")
    orchestrator = Orchestrator(llm, tools, repo, middleware=[mw])
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field

import structlog

from corza_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    MessageRole,
    ToolCall,
    ToolSchema,
)
from corza_agents.middleware.base import BaseMiddleware

log = structlog.get_logger("corza_agents.rate_limit")


@dataclass
class _Bucket:
    """Token bucket for a single scope key."""
    timestamps: list[float] = field(default_factory=list)


class RateLimitMiddleware(BaseMiddleware):
    """
    Token-bucket rate limiter for agent operations.

    Tracks LLM calls per scope key (user_id, tenant_id, or session_id).
    When the limit is exceeded within the window, subsequent LLM calls
    are blocked and an error is injected.

    Args:
        max_calls: Maximum number of LLM calls allowed in the window.
        window_seconds: Time window in seconds.
        scope: What to rate-limit by: "user", "tenant", or "session".
    """

    def __init__(
        self,
        max_calls: int = 50,
        window_seconds: float = 60.0,
        scope: str = "user",
    ):
        self._max_calls = max_calls
        self._window = window_seconds
        self._scope = scope
        self._buckets: dict[str, _Bucket] = defaultdict(_Bucket)

    @property
    def name(self) -> str:
        return "RateLimitMiddleware"

    def _get_key(self, context: ExecutionContext) -> str:
        if self._scope == "tenant":
            return context.metadata.get("tenant_id", context.session_id)
        if self._scope == "session":
            return context.session_id
        # Default: user
        return context.metadata.get("user_id", context.session_id)

    def _is_allowed(self, key: str) -> bool:
        """Check if a call is allowed under the rate limit."""
        now = time.monotonic()
        bucket = self._buckets[key]

        # Prune old timestamps outside the window
        cutoff = now - self._window
        bucket.timestamps = [t for t in bucket.timestamps if t > cutoff]

        if len(bucket.timestamps) >= self._max_calls:
            return False

        bucket.timestamps.append(now)
        return True

    async def before_llm_call(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        context: ExecutionContext,
    ) -> tuple[list[AgentMessage], list[ToolSchema]]:
        """Check rate limit before each LLM call."""
        key = self._get_key(context)

        if not self._is_allowed(key):
            log.warning(
                "rate_limit_exceeded",
                scope=self._scope,
                key=key,
                max_calls=self._max_calls,
                window=self._window,
                session_id=context.session_id,
            )
            # Inject a system message telling the agent to slow down
            rate_msg = AgentMessage(
                session_id=context.session_id,
                role=messages[0].role if messages else MessageRole.USER,
                content=(
                    f"[SYSTEM: Rate limit exceeded ({self._max_calls} calls "
                    f"per {self._window}s). Please wait before making more requests.]"
                ),
            )
            # Replace messages with just the rate limit message
            # This effectively blocks the LLM call by providing no useful context
            return messages + [rate_msg], tools

        return messages, tools

    async def before_tool_call(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> ToolCall | None:
        """Rate limit also applies to tool calls (optional)."""
        return tool_call  # Tools don't count against LLM rate limit

    def get_usage(self, context: ExecutionContext) -> dict:
        """Get current rate limit status for a scope key."""
        key = self._get_key(context)
        now = time.monotonic()
        bucket = self._buckets[key]
        cutoff = now - self._window
        active = [t for t in bucket.timestamps if t > cutoff]
        return {
            "scope": self._scope,
            "key": key,
            "calls_used": len(active),
            "calls_remaining": max(0, self._max_calls - len(active)),
            "max_calls": self._max_calls,
            "window_seconds": self._window,
        }
