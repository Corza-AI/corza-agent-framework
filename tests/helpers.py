"""
Shared test fixtures and mock helpers for corza-agents tests.
"""
from collections.abc import AsyncIterator

from corza_agents.core.types import (
    LLMStreamChunk,
    LLMUsage,
    StopReason,
    ToolCall,
)

# ══════════════════════════════════════════════════════════════════════
# Mock LLM helpers
# ══════════════════════════════════════════════════════════════════════

def make_text_response(text: str) -> list[LLMStreamChunk]:
    """Simulate an LLM that returns plain text (no tool calls)."""
    return [
        LLMStreamChunk(type="text_delta", text=text),
        LLMStreamChunk(
            type="usage",
            usage=LLMUsage(input_tokens=50, output_tokens=20, total_tokens=70),
            stop_reason=StopReason.END_TURN,
        ),
    ]


def make_tool_call_response(
    tool_name: str, arguments: dict, call_id: str = "tc-001"
) -> list[LLMStreamChunk]:
    """Simulate an LLM that requests a tool call."""
    return [
        LLMStreamChunk(type="text_delta", text="Let me look that up."),
        LLMStreamChunk(
            type="tool_call_end",
            tool_call=ToolCall(id=call_id, tool_name=tool_name, arguments=arguments),
        ),
        LLMStreamChunk(
            type="usage",
            usage=LLMUsage(input_tokens=60, output_tokens=30, total_tokens=90),
            stop_reason=StopReason.TOOL_USE,
        ),
    ]


class MockLLM:
    """LLM mock that returns pre-scripted responses in sequence."""

    def __init__(self, responses: list[list[LLMStreamChunk]]):
        self._responses = list(responses)
        self._call_count = 0

    async def stream_with_tools(self, **kwargs) -> AsyncIterator[LLMStreamChunk]:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        for chunk in self._responses[idx]:
            yield chunk

    async def count_tokens(self, text: str, model: str = "") -> int:
        return len(text.split())
