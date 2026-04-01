"""
Corza Agent Framework — LoopGuard Middleware

Detects and breaks infinite loops in the agent ReAct loop.
Three detection mechanisms:

1. Identical tool calls (same tool + args hash) repeated N times
2. Tool-only turns without substantive text output
3. Management-only turns (agent shuffling notes without real work)

On soft detection: injects a system message nudging the agent to wrap up.
On hard detection: sets _session_complete in working memory → engine stops.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import structlog

from corza_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    LLMResponse,
    MessageRole,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from corza_agents.middleware.base import BaseMiddleware

log = structlog.get_logger("corza_agents.loop_guard")


@dataclass
class LoopGuardConfig:
    """Configuration for loop detection thresholds."""

    # Identical call detection: same tool+args hash N times → force stop
    max_identical_calls: int = 3

    # Tool-only stagnation: N consecutive turns with tools but no text → intervene
    max_toolonly_turns: int = 4

    # Management-only: N turns with ONLY these tools (no substantive work)
    max_management_only_turns: int = 5
    management_tools: frozenset[str] = frozenset(
        {
            "manage_plan",
            "manage_notes",
            "manage_knowledge",
            "manage_objective",
            "manage_context",
            "manage_skill",
        }
    )

    # Give one soft warning before hard stopping
    soft_warning_before_hard_stop: bool = True


class LoopGuardMiddleware(BaseMiddleware):
    """
    Detects and breaks infinite loops in the agent ReAct loop.

    Usage::

        from corza_agents.middleware.loop_guard import LoopGuardMiddleware

        engine = AgentEngine(
            llm=llm, tool_registry=registry, repository=repo,
            middleware=[LoopGuardMiddleware()],
        )

    With custom config::

        config = LoopGuardConfig(max_identical_calls=5, max_toolonly_turns=12)
        middleware = [LoopGuardMiddleware(config=config)]
    """

    def __init__(self, config: LoopGuardConfig | None = None):
        self._config = config or LoopGuardConfig()
        # Per-session tracking (keyed by session_id)
        self._call_hashes: dict[str, list[str]] = {}
        self._toolonly_turns: dict[str, int] = {}
        self._mgmt_only_turns: dict[str, int] = {}
        self._warned: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "LoopGuard"

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _hash_call(tc: ToolCall) -> str:
        """Stable hash of tool_name + arguments for dedup detection."""
        raw = f"{tc.tool_name}:{json.dumps(tc.arguments, sort_keys=True, default=str)}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _soft_warn(self, context: ExecutionContext, reason: str) -> None:
        """Store a warning in working memory. Injected on next LLM call."""
        log.info("loop_guard_soft_warning", session_id=context.session_id, reason=reason)
        if context.working_memory:
            context.working_memory.store(
                "_loop_guard_warning",
                f"[LOOP GUARD] {reason}. "
                "You MUST wrap up NOW. Deliver your final response and call "
                "session_complete(summary='...'). Do NOT call more tools.",
            )

    def _force_stop(self, context: ExecutionContext, reason: str) -> None:
        """Force session termination via working memory flag."""
        log.warning("loop_guard_force_stop", session_id=context.session_id, reason=reason)
        if context.working_memory:
            context.working_memory.store("_session_complete", True)
            context.working_memory.store(
                "_session_summary",
                f"Session terminated by LoopGuard: {reason}",
            )

    def _intervene(self, context: ExecutionContext, reason: str) -> None:
        """Soft warn first, then hard stop on second violation."""
        sid = context.session_id
        if self._config.soft_warning_before_hard_stop and not self._warned.get(sid):
            self._soft_warn(context, reason)
            self._warned[sid] = True
        else:
            self._force_stop(context, reason)

    def _cleanup(self, sid: str) -> None:
        """Remove tracking state for a completed session."""
        self._call_hashes.pop(sid, None)
        self._toolonly_turns.pop(sid, None)
        self._mgmt_only_turns.pop(sid, None)
        self._warned.pop(sid, None)

    # ── Middleware hooks ────────────────────────────────────────────

    async def before_llm_call(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        context: ExecutionContext,
    ) -> tuple[list[AgentMessage], list[ToolSchema]]:
        """Inject loop guard warning into messages if one is pending."""
        wm = context.working_memory
        if wm:
            warning = wm.get("_loop_guard_warning")
            if warning:
                wm.remove("_loop_guard_warning")
                warning_msg = AgentMessage(
                    session_id=context.session_id,
                    role=MessageRole.USER,
                    content=f"[SYSTEM] {warning}",
                )
                messages = list(messages) + [warning_msg]
        return messages, tools

    async def after_llm_call(
        self,
        response: LLMResponse,
        context: ExecutionContext,
    ) -> LLMResponse:
        """Track whether LLM produced text and management-only tool calls."""
        wm = context.working_memory
        sid = context.session_id
        if not wm:
            return response

        has_text = bool(response.content and response.content.strip())

        # Track management-only turns (no text + only management tools)
        if response.tool_calls:
            all_mgmt = all(
                tc.tool_name in self._config.management_tools for tc in response.tool_calls
            )
            if all_mgmt and not has_text:
                self._mgmt_only_turns[sid] = self._mgmt_only_turns.get(sid, 0) + 1
            else:
                self._mgmt_only_turns[sid] = 0

            if self._mgmt_only_turns.get(sid, 0) >= self._config.max_management_only_turns:
                self._intervene(
                    context,
                    f"Only management tools for {self._config.max_management_only_turns} "
                    f"consecutive turns with no substantive work",
                )

        # Store text flag for on_turn_complete stagnation check
        wm.store("_loop_guard_last_had_text", has_text)

        return response

    async def after_tool_call(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        context: ExecutionContext,
    ) -> ToolResult:
        """Track tool call signatures for identical-call detection."""
        sid = context.session_id
        if sid not in self._call_hashes:
            self._call_hashes[sid] = []

        h = self._hash_call(tool_call)
        self._call_hashes[sid].append(h)

        # Check for identical calls in recent history
        n = self._config.max_identical_calls
        recent = self._call_hashes[sid][-n:]
        if len(recent) == n and len(set(recent)) == 1:
            log.warning(
                "loop_guard_identical_calls", session_id=sid, tool=tool_call.tool_name, count=n
            )
            self._force_stop(
                context,
                f"Identical tool call detected: {tool_call.tool_name} "
                f"called {n} times with same arguments",
            )

        return result

    async def on_turn_complete(
        self,
        turn_number: int,
        context: ExecutionContext,
    ) -> None:
        """Track tool-only turns for stagnation detection."""
        sid = context.session_id
        wm = context.working_memory
        if not wm:
            return

        last_had_text = wm.get("_loop_guard_last_had_text")

        # Only count as tool-only if the flag was explicitly False
        # (None means text-only turn which is fine)
        if last_had_text is False:
            self._toolonly_turns[sid] = self._toolonly_turns.get(sid, 0) + 1
        else:
            self._toolonly_turns[sid] = 0

        if self._toolonly_turns.get(sid, 0) >= self._config.max_toolonly_turns:
            self._intervene(
                context,
                f"No text output for {self._config.max_toolonly_turns} "
                f"consecutive turns — agent is looping without communicating",
            )
