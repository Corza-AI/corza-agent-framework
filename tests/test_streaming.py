"""Tests for streaming events."""
from corza_agents.core.types import EventType
from corza_agents.streaming.events import (
    error_event,
    session_completed,
    session_started,
    text_delta,
    tool_executing,
    tool_result_event,
    turn_completed,
)


def test_session_started():
    event = session_started("s1", "agent_1", "brain")
    assert event.type == EventType.SESSION_STARTED
    assert event.session_id == "s1"
    assert event.data["agent_name"] == "brain"


def test_text_delta():
    event = text_delta("s1", "Hello", turn=1)
    assert event.type == EventType.LLM_TEXT_DELTA
    assert event.data["text"] == "Hello"
    assert event.turn_number == 1


def test_tool_executing():
    event = tool_executing("s1", "search", "tc1", turn=2)
    assert event.data["tool_name"] == "search"


def test_tool_result():
    event = tool_result_event("s1", "search", "tc1", "success", 42.0, "preview", turn=2)
    assert event.data["status"] == "success"
    assert event.data["duration_ms"] == 42.0


def test_turn_completed():
    event = turn_completed("s1", 3, "end_turn")
    assert event.data["stop_reason"] == "end_turn"


def test_session_completed():
    event = session_completed("s1", 5, 1000, 500, "Final output")
    assert event.data["total_turns"] == 5
    assert event.data["final_output"] == "Final output"


def test_error_event():
    event = error_event("s1", "llm_error", "timeout", turn=1, recoverable=True)
    assert event.type == EventType.ERROR
    assert event.data["recoverable"] is True


def test_to_sse():
    event = text_delta("s1", "hi")
    sse = event.to_sse()
    assert "id: " in sse  # Event ID for reconnection
    assert "event: llm.text_delta\n" in sse
    assert "data:" in sse
    assert sse.endswith("\n\n")


def test_to_dict():
    event = session_started("s1", "a1", "brain")
    d = event.to_dict()
    assert d["session_id"] == "s1"
    assert d["type"] == "session.started"
