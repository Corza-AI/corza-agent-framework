"""
Corza Agent Framework — Audit Middleware

Logs every LLM call and tool execution to the audit table.
Provides full traceability for compliance and debugging.
"""

import json

import structlog

from corza_agents.core.types import (
    ExecutionContext,
    LLMResponse,
    ToolCall,
    ToolResult,
)
from corza_agents.middleware.base import BaseMiddleware
from corza_agents.persistence.base import BaseRepository

log = structlog.get_logger("corza_agents.audit")


class AuditMiddleware(BaseMiddleware):
    """
    Logs every significant action to the af_audit_log table.

    Captures:
    - LLM calls (model, tokens, latency)
    - Tool executions (name, status, duration)
    - Errors
    - Turn completions
    """

    def __init__(self, repository: BaseRepository):
        self._repo = repository

    @property
    def name(self) -> str:
        return "AuditMiddleware"

    async def after_llm_call(
        self,
        response: LLMResponse,
        context: ExecutionContext,
    ) -> LLMResponse:
        tool_names = [tc.tool_name for tc in response.tool_calls] if response.tool_calls else []
        await self._repo.log_audit(
            session_id=context.session_id,
            event_type="llm_call",
            actor=context.agent_name,
            action=f"LLM call → {response.model} (stop={response.stop_reason.value})"
            + (f", tools=[{', '.join(tool_names)}]" if tool_names else ""),
            detail={
                "model": response.model,
                "stop_reason": response.stop_reason.value,
                "tool_calls": tool_names,
                "content_preview": response.content[:200] if response.content else "",
            },
            llm_model=response.model,
            llm_input_tokens=response.usage.input_tokens,
            llm_output_tokens=response.usage.output_tokens,
            llm_latency_ms=response.latency_ms,
        )
        return response

    async def after_tool_call(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        context: ExecutionContext,
    ) -> ToolResult:
        output_preview = ""
        if result.output:
            if isinstance(result.output, str):
                output_preview = result.output[:200]
            else:
                output_preview = json.dumps(result.output, default=str)[:200]

        await self._repo.log_audit(
            session_id=context.session_id,
            event_type="tool_execution",
            actor=context.agent_name,
            action=f"Tool '{tool_call.tool_name}' → {result.status.value} ({result.duration_ms:.0f}ms)",
            detail={
                "tool_name": tool_call.tool_name,
                "tool_call_id": tool_call.id,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "input_keys": list(tool_call.arguments.keys()),
                "output_preview": output_preview,
                "error": result.error,
            },
        )
        return result

    async def on_error(self, error: Exception, context: ExecutionContext) -> None:
        await self._repo.log_audit(
            session_id=context.session_id,
            event_type="error",
            actor=context.agent_name,
            action=f"Error: {type(error).__name__}: {str(error)[:500]}",
            detail={
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                "turn": context.turn_number,
            },
        )

    async def on_turn_complete(self, turn_number: int, context: ExecutionContext) -> None:
        await self._repo.log_audit(
            session_id=context.session_id,
            event_type="turn_complete",
            actor=context.agent_name,
            action=f"Turn {turn_number} completed",
        )
