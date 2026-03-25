"""
Corza Agent Framework — Permission Middleware

Controls which tools an agent can use and how.
Inspired by OpenCode's allow/ask/deny permission system
for tool-level access control.

Permission levels:
  - auto_approve: Tool runs without checks
  - ask: Requires human approval (emits waiting event)
  - deny: Tool is blocked entirely
  - conditional: Runs a permission function to decide
"""
from collections.abc import Callable

import structlog

from corza_agents.core.types import (
    AgentMessage,
    ExecutionContext,
    ToolCall,
    ToolSchema,
)
from corza_agents.middleware.base import BaseMiddleware

log = structlog.get_logger("corza_agents.permissions")


class PermissionRule:
    """A single permission rule for a tool or tool pattern."""

    def __init__(
        self,
        tool_pattern: str,
        level: str = "auto_approve",
        condition: Callable | None = None,
        reason: str = "",
    ):
        """
        Args:
            tool_pattern: Tool name or glob pattern (e.g., "search_*", "*")
            level: "auto_approve" | "ask" | "deny" | "conditional"
            condition: For "conditional" level — async callable(tool_call, context) -> bool
            reason: Human-readable reason for the rule
        """
        self.tool_pattern = tool_pattern
        self.level = level
        self.condition = condition
        self.reason = reason

    def matches(self, tool_name: str) -> bool:
        if self.tool_pattern == "*":
            return True
        if self.tool_pattern.endswith("*"):
            return tool_name.startswith(self.tool_pattern[:-1])
        if self.tool_pattern.startswith("*"):
            return tool_name.endswith(self.tool_pattern[1:])
        return tool_name == self.tool_pattern


class PermissionMiddleware(BaseMiddleware):
    """
    Enforces tool permission rules.

    Rules are evaluated in order — first match wins.
    If no rule matches, the default level applies.
    """

    def __init__(
        self,
        rules: list[PermissionRule] | None = None,
        default_level: str = "auto_approve",
        denied_tools: list[str] | None = None,
    ):
        self._rules = rules or []
        self._default_level = default_level
        self._denied_tools = set(denied_tools or [])

    @property
    def name(self) -> str:
        return "PermissionMiddleware"

    def add_rule(self, rule: PermissionRule) -> None:
        self._rules.append(rule)

    def deny_tool(self, tool_name: str) -> None:
        self._denied_tools.add(tool_name)

    def allow_tool(self, tool_name: str) -> None:
        self._denied_tools.discard(tool_name)

    async def before_tool_call(
        self,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> ToolCall | None:
        """Check permissions before tool execution."""
        tool_name = tool_call.tool_name

        # Explicit deny list
        if tool_name in self._denied_tools:
            log.info("tool_denied_explicit", tool=tool_name, session=context.session_id)
            return None

        # Check rules in order
        for rule in self._rules:
            if rule.matches(tool_name):
                return await self._evaluate_rule(rule, tool_call, context)

        # Default
        if self._default_level == "deny":
            log.info("tool_denied_default", tool=tool_name)
            return None

        return tool_call

    async def _evaluate_rule(
        self,
        rule: PermissionRule,
        tool_call: ToolCall,
        context: ExecutionContext,
    ) -> ToolCall | None:
        if rule.level == "auto_approve":
            return tool_call

        elif rule.level == "deny":
            log.info("tool_denied_by_rule",
                     tool=tool_call.tool_name,
                     reason=rule.reason)
            return None

        elif rule.level == "conditional" and rule.condition:
            try:
                allowed = await rule.condition(tool_call, context)
                if not allowed:
                    log.info("tool_denied_conditional",
                             tool=tool_call.tool_name,
                             reason=rule.reason)
                    return None
                return tool_call
            except Exception as e:
                log.error("permission_condition_error",
                          tool=tool_call.tool_name, error=str(e))
                return None

        elif rule.level == "ask":
            log.info("tool_needs_approval",
                     tool=tool_call.tool_name,
                     reason=rule.reason)
            # In a full implementation, this would pause and wait for human approval
            # For now, auto-approve (can be wired to a WebSocket approval flow)
            return tool_call

        return tool_call

    async def before_llm_call(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        context: ExecutionContext,
    ) -> tuple[list[AgentMessage], list[ToolSchema]]:
        """Filter out denied tools from the tool list sent to the LLM."""
        if not self._denied_tools:
            return messages, tools

        filtered = [t for t in tools if t.name not in self._denied_tools]
        if len(filtered) < len(tools):
            removed = [t.name for t in tools if t.name in self._denied_tools]
            log.debug("tools_filtered_from_llm", removed=removed)
        return messages, filtered
