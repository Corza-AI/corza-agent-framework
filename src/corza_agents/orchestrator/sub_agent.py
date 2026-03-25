"""
Corza Agent Framework — Sub-Agent Runner

Full lifecycle management for sub-agents:
- Spawn: create a new sub-agent with a task
- Message: send follow-up messages to an existing sub-agent
- Status: check if a sub-agent is still running or completed

Each sub-agent gets its own DB session, linked to the parent via
parent_session_id. Sessions persist — the orchestrator can return
to a sub-agent across multiple turns.
"""
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from corza_agents.core.engine import AgentEngine
from corza_agents.core.types import (
    AgentDefinition,
    AgentSession,
    LLMUsage,
    MessageRole,
    SubAgentResult,
    ToolStatus,
    _uuid,
)
from corza_agents.streaming.events import StreamEvent, subagent_completed, subagent_started

log = structlog.get_logger("corza_agents.sub_agent")

EventCallback = Callable[[StreamEvent], Awaitable[None]]


class SubAgentRunner:
    """
    Spawn and manage sub-agents with isolated context.

    Design:
    - Each sub-agent creates a CHILD session in DB (linked via parent_session_id)
    - Sub-agent gets a fresh message history — only the task description + optional context
    - Runs its own ReAct loop via the shared AgentEngine
    - Returns ONLY the final result to the parent (full conversation persisted in child)
    - All intermediate reasoning stays isolated (auditable via child session)
    - Optional on_event callback streams events in real-time (e.g. to WebSocket)
    """

    def __init__(self, engine: AgentEngine):
        self._engine = engine

    async def run(
        self,
        task: str,
        agent_def: AgentDefinition,
        parent_session_id: str,
        context_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        on_event: EventCallback | None = None,
    ) -> SubAgentResult:
        """
        Run a sub-agent to completion and return its result.

        Args:
            task: Clear description of what the sub-agent should do
            agent_def: The sub-agent's definition (model, tools, skills, prompt)
            parent_session_id: Parent session for linking
            context_data: Optional data from parent to pass as context
            metadata: Additional metadata for the child session
            on_event: Optional async callback invoked for every stream event,
                      enabling real-time forwarding to WebSocket / UI.

        Returns:
            SubAgentResult with the final output text, data, and usage stats
        """
        child_session_id = _uuid()

        child_metadata = {
            "parent_session_id": parent_session_id,
            "task_preview": task[:200],
            **(metadata or {}),
        }

        user_message = self._build_task_message(task, context_data)

        log.info("subagent_start",
                 child_session_id=child_session_id,
                 parent_session_id=parent_session_id,
                 agent_name=agent_def.name,
                 task_preview=task[:100])

        # Strip non-serializable objects (like LLM instances) for DB persistence
        db_metadata = {
            k: v for k, v in child_metadata.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        child_session = AgentSession(
            id=child_session_id,
            agent_id=agent_def.id,
            parent_session_id=parent_session_id,
            metadata=db_metadata,
        )
        await self._engine.repository.create_session(child_session)

        final_text = ""
        total_usage = LLMUsage()
        turn_count = 0
        error_msg: str | None = None

        # Emit subagent.started event
        if on_event:
            try:
                await on_event(subagent_started(
                    parent_session_id, child_session_id, agent_def.name, task[:200],
                ))
            except Exception:
                pass

        try:
            async for event in self._engine.run(
                session_id=child_session_id,
                user_message=user_message,
                agent_def=agent_def,
                metadata=child_metadata,
            ):
                if on_event:
                    try:
                        await on_event(event)
                    except Exception as cb_err:
                        log.warning("on_event_callback_error", error=str(cb_err))

                if event.type.value == "session.completed":
                    final_text = event.data.get("final_output", "")
                    total_usage.input_tokens = event.data.get("input_tokens", 0)
                    total_usage.output_tokens = event.data.get("output_tokens", 0)
                    turn_count = event.data.get("total_turns", 0)
                elif event.type.value == "error":
                    error_msg = event.data.get("message", "Unknown error")

        except Exception as e:
            log.error("subagent_error",
                      child_session_id=child_session_id,
                      error=str(e)[:500])
            error_msg = str(e)

        # Robust result capture — 3-tier fallback
        if not final_text:
            messages = await self._engine.repository.get_messages(child_session_id)

            # Tier 1: last assistant message with text content (CoT reasoning)
            for m in reversed(messages):
                if m.role == MessageRole.ASSISTANT and m.content:
                    text = m.text() if isinstance(m.content, list) else str(m.content)
                    if text and text.strip():
                        final_text = text.strip()
                        break

            # Tier 2: compile from tool results (the actual work product)
            if not final_text:
                tool_outputs = []
                for m in reversed(messages):
                    if m.role == MessageRole.TOOL_RESULT and m.content:
                        content = m.content if isinstance(m.content, str) else str(m.content)
                        if content.strip() and len(content.strip()) > 20:
                            label = m.tool_name or "tool"
                            tool_outputs.append(f"[{label}] {content.strip()[:3000]}")
                            if len(tool_outputs) >= 10:
                                break
                if tool_outputs:
                    tool_outputs.reverse()
                    final_text = "Sub-agent results:\n\n" + "\n\n".join(tool_outputs)

            # Tier 3: minimal summary so the orchestrator knows something happened
            if not final_text:
                final_text = (
                    f"Sub-agent '{agent_def.name}' completed {turn_count} turns. "
                    f"No text output was produced. Check child session {child_session_id} for details."
                )

        log.info("subagent_complete",
                 child_session_id=child_session_id,
                 turns=turn_count,
                 tokens=total_usage.total_tokens,
                 output_length=len(final_text),
                 has_error=bool(error_msg))

        # If we have useful output, treat as success even if there was an error
        # (the error details are appended to the output so the orchestrator can see them)
        has_useful_output = bool(final_text and len(final_text) > 50)
        if error_msg and has_useful_output:
            final_text += f"\n\n[Note: sub-agent encountered an error: {error_msg}]"

        # Emit subagent.completed event
        status_val = "success" if has_useful_output else ("error" if error_msg else "success")
        if on_event:
            try:
                await on_event(subagent_completed(
                    parent_session_id, child_session_id, agent_def.name,
                    status_val, turn_count,
                ))
            except Exception:
                pass

        return SubAgentResult(
            child_session_id=child_session_id,
            output=final_text,
            data={"task": task},
            status=ToolStatus.SUCCESS if has_useful_output else (ToolStatus.ERROR if error_msg else ToolStatus.SUCCESS),
            error=error_msg if not has_useful_output else None,
            turns_used=turn_count,
            tokens_used=total_usage,
        )

    async def send_message(
        self,
        child_session_id: str,
        message: str,
        agent_def: AgentDefinition,
        metadata: dict[str, Any] | None = None,
        on_event: EventCallback | None = None,
    ) -> SubAgentResult:
        """
        Send a follow-up message to an existing sub-agent session.

        The sub-agent resumes its conversation with full history intact.
        Use this to refine, redirect, or ask follow-up questions.

        Args:
            child_session_id: The session ID of the sub-agent to message
            message: The follow-up instruction
            agent_def: The sub-agent's definition (for model/tools config)
            metadata: Additional metadata
            on_event: Optional event callback

        Returns:
            SubAgentResult with the new response
        """
        log.info("subagent_followup",
                 child_session_id=child_session_id,
                 message_preview=message[:100])

        final_text = ""
        total_usage = LLMUsage()
        turn_count = 0
        error_msg: str | None = None

        try:
            async for event in self._engine.run(
                session_id=child_session_id,
                user_message=message,
                agent_def=agent_def,
                metadata=metadata or {},
            ):
                if on_event:
                    try:
                        await on_event(event)
                    except Exception as cb_err:
                        log.warning("on_event_callback_error", error=str(cb_err))

                if event.type.value == "session.completed":
                    final_text = event.data.get("final_output", "")
                    total_usage.input_tokens = event.data.get("input_tokens", 0)
                    total_usage.output_tokens = event.data.get("output_tokens", 0)
                    turn_count = event.data.get("total_turns", 0)
                elif event.type.value == "error":
                    error_msg = event.data.get("message", "Unknown error")

        except Exception as e:
            log.error("subagent_followup_error",
                      child_session_id=child_session_id, error=str(e)[:500])
            error_msg = str(e)

        # Same 3-tier fallback as run()
        if not final_text:
            messages = await self._engine.repository.get_messages(child_session_id)
            for m in reversed(messages):
                if m.role == MessageRole.ASSISTANT and m.content:
                    text = m.text() if isinstance(m.content, list) else str(m.content)
                    if text and text.strip():
                        final_text = text.strip()
                        break

        if not final_text:
            final_text = f"Sub-agent follow-up completed {turn_count} turns with no text output."

        return SubAgentResult(
            child_session_id=child_session_id,
            output=final_text,
            data={"follow_up": message[:200]},
            status=ToolStatus.SUCCESS if final_text else ToolStatus.ERROR,
            error=error_msg,
            turns_used=turn_count,
            tokens_used=total_usage,
        )

    @staticmethod
    def _build_task_message(task: str, context_data: dict | None = None) -> str:
        """Build the initial message for the sub-agent."""
        parts = [task]
        if context_data:
            parts.append("\n## Context Data\n")
            for key, value in context_data.items():
                if isinstance(value, (dict, list)):
                    val_str = json.dumps(value, indent=2, default=str)
                    if len(val_str) > 3000:
                        val_str = val_str[:3000] + "\n... (truncated)"
                else:
                    val_str = str(value)
                parts.append(f"**{key}**:\n{val_str}\n")
        return "\n".join(parts)
