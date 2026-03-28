"""
Corza Agent Framework — Middleware Protocol

Middleware hooks into the agent loop at defined points.
Middleware hook system for agent lifecycle events.

Middleware is applied in order: first registered = first to run.
Each method receives the current state and can modify/inspect it.
"""

from typing import Protocol, runtime_checkable

from corza_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    LLMResponse,
    ToolCall,
    ToolResult,
    ToolSchema,
)


@runtime_checkable
class Middleware(Protocol):
    """
    Protocol for agent loop middleware.

    Implement any subset of these methods. Unimplemented methods
    default to pass-through (return inputs unchanged).
    """

    @property
    def name(self) -> str:
        """Unique middleware name for logging and error attribution."""
        ...

    async def before_llm_call(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        context: ExecutionContext,
    ) -> tuple[list[AgentMessage], list[ToolSchema]]:
        """
        Called before each LLM invocation.
        Can modify messages (e.g., inject context) or tools (e.g., filter by permission).
        """
        ...

    async def after_llm_call(
        self,
        response: LLMResponse,
        context: ExecutionContext,
    ) -> LLMResponse:
        """Called after LLM response is fully received. Can inspect/modify."""
        ...

    async def before_tool_call(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> ToolCall | None:
        """
        Called before each tool execution.
        Return the (possibly modified) tool_call, or None to BLOCK execution.
        """
        ...

    async def after_tool_call(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        context: ExecutionContext,
    ) -> ToolResult:
        """Called after tool execution. Can inspect/modify result."""
        ...

    async def on_turn_complete(
        self,
        turn_number: int,
        context: ExecutionContext,
    ) -> None:
        """Called at the end of each ReAct turn."""
        ...

    async def on_error(
        self,
        error: Exception,
        context: ExecutionContext,
    ) -> None:
        """Called when an error occurs in the agent loop."""
        ...


class BaseMiddleware:
    """
    Convenience base class implementing pass-through for all hooks.
    Subclass and override only the methods you need.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def before_llm_call(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        context: ExecutionContext,
    ) -> tuple[list[AgentMessage], list[ToolSchema]]:
        return messages, tools

    async def after_llm_call(
        self,
        response: LLMResponse,
        context: ExecutionContext,
    ) -> LLMResponse:
        return response

    async def before_tool_call(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> ToolCall | None:
        return tool_call

    async def after_tool_call(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        context: ExecutionContext,
    ) -> ToolResult:
        return result

    async def on_turn_complete(self, turn_number: int, context: ExecutionContext) -> None:
        pass

    async def on_error(self, error: Exception, context: ExecutionContext) -> None:
        pass
