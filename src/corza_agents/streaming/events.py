"""
Corza Agent Framework — Streaming Events

Real-time events emitted during agent execution.
Consumed by SSE/WebSocket API endpoints to stream progress to clients.
"""
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from corza_agents.core.types import EventType


def _now() -> datetime:
    return datetime.now(UTC)


def _event_id() -> str:
    return str(uuid.uuid4())


class StreamEvent(BaseModel):
    """
    A single event emitted during agent execution.

    Events flow: Engine → AsyncIterator → SSE endpoint → Frontend
    """
    id: str = Field(default_factory=_event_id)
    type: EventType
    session_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)
    turn_number: int | None = None

    def to_sse(self) -> str:
        """Format as Server-Sent Event with ID for reconnection support."""
        payload = self.model_dump(mode="json")
        return f"id: {self.id}\nevent: {self.type.value}\ndata: {json.dumps(payload)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ══════════════════════════════════════════════════════════════════════
# Event Constructors — convenience functions for common events
# ══════════════════════════════════════════════════════════════════════

def session_started(session_id: str, agent_id: str, agent_name: str) -> StreamEvent:
    return StreamEvent(
        type=EventType.SESSION_STARTED,
        session_id=session_id,
        data={"agent_id": agent_id, "agent_name": agent_name},
    )


def turn_started(session_id: str, turn: int) -> StreamEvent:
    return StreamEvent(
        type=EventType.TURN_STARTED,
        session_id=session_id,
        turn_number=turn,
        data={"turn": turn},
    )


def text_delta(session_id: str, text: str, turn: int = 0) -> StreamEvent:
    return StreamEvent(
        type=EventType.LLM_TEXT_DELTA,
        session_id=session_id,
        turn_number=turn,
        data={"text": text},
    )


def thinking_delta(session_id: str, text: str, turn: int = 0) -> StreamEvent:
    return StreamEvent(
        type=EventType.LLM_THINKING,
        session_id=session_id,
        turn_number=turn,
        data={"text": text},
    )


def tool_call_event(session_id: str, tool_name: str, tool_call_id: str,
                    arguments: dict, turn: int = 0) -> StreamEvent:
    return StreamEvent(
        type=EventType.LLM_TOOL_CALL,
        session_id=session_id,
        turn_number=turn,
        data={
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "arguments": arguments,
        },
    )


def tool_executing(session_id: str, tool_name: str, tool_call_id: str,
                   turn: int = 0) -> StreamEvent:
    return StreamEvent(
        type=EventType.TOOL_EXECUTING,
        session_id=session_id,
        turn_number=turn,
        data={"tool_name": tool_name, "tool_call_id": tool_call_id},
    )


def tool_result_event(session_id: str, tool_name: str, tool_call_id: str,
                      status: str, duration_ms: float, output_preview: str = "",
                      turn: int = 0,
                      card_data: dict | None = None) -> StreamEvent:
    data = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "status": status,
        "duration_ms": duration_ms,
        "output_preview": output_preview,
    }
    if card_data:
        data["card_data"] = card_data
    return StreamEvent(
        type=EventType.TOOL_RESULT,
        session_id=session_id,
        turn_number=turn,
        data=data,
    )


def subagent_started(session_id: str, child_session_id: str,
                     agent_name: str, task: str) -> StreamEvent:
    return StreamEvent(
        type=EventType.SUBAGENT_STARTED,
        session_id=session_id,
        data={
            "child_session_id": child_session_id,
            "agent_name": agent_name,
            "task": task[:200],
        },
    )


def subagent_completed(session_id: str, child_session_id: str,
                       agent_name: str, status: str) -> StreamEvent:
    return StreamEvent(
        type=EventType.SUBAGENT_COMPLETED,
        session_id=session_id,
        data={
            "child_session_id": child_session_id,
            "agent_name": agent_name,
            "status": status,
        },
    )


def context_compacting(session_id: str, messages_before: int,
                       messages_after: int) -> StreamEvent:
    return StreamEvent(
        type=EventType.CONTEXT_COMPACTING,
        session_id=session_id,
        data={
            "messages_before": messages_before,
            "messages_after": messages_after,
        },
    )


def turn_completed(session_id: str, turn: int, stop_reason: str) -> StreamEvent:
    return StreamEvent(
        type=EventType.TURN_COMPLETED,
        session_id=session_id,
        turn_number=turn,
        data={"turn": turn, "stop_reason": stop_reason},
    )


def session_completed(session_id: str, total_turns: int,
                      input_tokens: int, output_tokens: int,
                      final_output: str = "") -> StreamEvent:
    return StreamEvent(
        type=EventType.SESSION_COMPLETED,
        session_id=session_id,
        data={
            "total_turns": total_turns,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "final_output": final_output[:2000],
        },
    )


def error_event(session_id: str, error_type: str, message: str,
                turn: int = 0, recoverable: bool = False) -> StreamEvent:
    return StreamEvent(
        type=EventType.ERROR,
        session_id=session_id,
        turn_number=turn,
        data={
            "error_type": error_type,
            "message": message[:1000],
            "recoverable": recoverable,
        },
    )
