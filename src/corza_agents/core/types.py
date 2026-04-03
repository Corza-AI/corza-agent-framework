"""
Corza Agent Framework — Core Types

All Pydantic models for the framework. These are the fundamental types
that every other module builds on.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED = "denied"


class ToolType(str, Enum):
    FUNCTION = "function"
    API = "api"
    DB_QUERY = "db_query"
    AGENT = "agent"
    WORKFLOW = "workflow"
    CODE = "code"


class EventType(str, Enum):
    SESSION_STARTED = "session.started"
    TURN_STARTED = "turn.started"
    LLM_THINKING = "llm.thinking"
    LLM_TEXT_DELTA = "llm.text_delta"
    LLM_TOOL_CALL = "llm.tool_call"
    LLM_RESPONSE_COMPLETE = "llm.response_complete"
    TOOL_EXECUTING = "tool.executing"
    TOOL_RESULT = "tool.result"
    SUBAGENT_STARTED = "subagent.started"
    SUBAGENT_COMPLETED = "subagent.completed"
    CONTEXT_COMPACTING = "context.compacting"
    TURN_COMPLETED = "turn.completed"
    SESSION_COMPLETED = "session.completed"
    ERROR = "error"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TURNS = "max_turns"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    ERROR = "error"
    CANCELLED = "cancelled"


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# ══════════════════════════════════════════════════════════════════════
# Tool Models
# ══════════════════════════════════════════════════════════════════════


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM."""

    id: str = Field(default_factory=_uuid)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The result of executing a tool."""

    tool_call_id: str
    tool_name: str
    output: Any = None
    status: ToolStatus = ToolStatus.SUCCESS
    duration_ms: float = 0.0
    error: str | None = None


class ToolParameterSpec(BaseModel):
    """Schema for a single tool parameter."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any | None = None
    enum: list[str] | None = None


class ToolSchema(BaseModel):
    """
    LLM-facing tool definition — the JSON Schema the model sees.
    Provider adapters convert this to Anthropic/OpenAI/etc format.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class RegisteredTool(BaseModel):
    """A tool registered in the ToolRegistry."""

    name: str
    description: str
    tool_type: ToolType = ToolType.FUNCTION
    parameters: list[ToolParameterSpec] = Field(default_factory=list)
    json_schema: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    requires_auth: bool = False
    permission_level: str = "auto_approve"
    timeout_seconds: int = 30
    retry_max: int = 0
    tags: list[str] = Field(default_factory=list)

    handler: Any | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def to_tool_schema(self) -> ToolSchema:
        """Convert to LLM-facing schema."""
        if self.json_schema:
            return ToolSchema(
                name=self.name,
                description=self.description,
                parameters=self.json_schema,
            )
        props = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            props[p.name] = prop
            if p.required:
                required.append(p.name)
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": props,
                "required": required,
            },
        )


# ══════════════════════════════════════════════════════════════════════
# Message Models
# ══════════════════════════════════════════════════════════════════════


class AgentMessage(BaseModel):
    """A single message in the conversation."""

    id: str = Field(default_factory=_uuid)
    session_id: str = ""
    role: MessageRole
    content: str | list[dict[str, Any]] = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    token_count: int = 0
    model: str | None = None
    created_at: datetime = Field(default_factory=_now)
    is_summarized: bool = False

    def text(self) -> str:
        """Extract plain text from content."""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Session Models
# ══════════════════════════════════════════════════════════════════════


class UserContext(BaseModel):
    """
    User/tenant context passed through from the web app.

    The framework never handles auth — your app does. This is the
    pass-through context that scopes sessions, messages, and memory
    to the right user and tenant.
    """

    user_id: str = ""
    tenant_id: str = ""
    org_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSession(BaseModel):
    """Top-level container for an agent run, scoped to a user and tenant."""

    id: str = Field(default_factory=_uuid)
    agent_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    parent_session_id: str | None = None
    status: SessionStatus = SessionStatus.IDLE
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    turn_count: int = 0
    error: str | None = None


# ══════════════════════════════════════════════════════════════════════
# Agent Definition
# ══════════════════════════════════════════════════════════════════════


class AgentDefinition(BaseModel):
    """
    Declarative agent configuration.

    Defines everything an agent needs: model, tools, skills, prompts,
    knowledge files, sub-agents it can spawn, and constraints.
    """

    id: str = Field(default_factory=_uuid)
    name: str
    description: str = ""
    objective: str = Field(
        default="",
        description="The agent's mission / purpose / north star (like CLAUDE.md or Soul.md). "
        "Auto-loaded into the system prompt every turn. "
        "Used as the initial seed — the agent can refine it at runtime via manage_objective.",
    )
    system_prompt: str = ""
    model: str = ""
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Fallback models tried in order if primary fails. "
        "E.g. ['groq:llama-3.3-70b', 'cerebras:llama-3.3-70b']",
    )
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    knowledge: list[Any] = Field(default_factory=list)  # str | dict | Callable
    max_turns: int = 50
    max_llm_retries: int = 3
    min_turns: int = 0
    max_tokens_per_turn: int = 8192
    llm_timeout_seconds: float = 120.0
    temperature: float = 0.0
    middleware: list[str] = Field(default_factory=list)
    sub_agents: dict[str, "AgentDefinition"] = Field(default_factory=dict)
    max_parallel_agents: int = Field(
        default=5,
        description="Max sub-agents to run concurrently via spawn_parallel. 1-10.",
    )
    memory_scope: str = "isolated"
    extra_context: str = Field(
        default="",
        description="Application-specific context appended to the system prompt "
        "(e.g. available data connections, workspace info). "
        "Placed after Plan, before the user's first message.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


# ══════════════════════════════════════════════════════════════════════
# Skill
# ══════════════════════════════════════════════════════════════════════


class Skill(BaseModel):
    """
    Reusable agent capability = prompt template + required tools + config.

    Skills are injected into the system prompt when activated.
    They can define which tools are needed, and provide Jinja2 templates
    with variables resolved at runtime.
    """

    id: str = Field(default_factory=_uuid)
    name: str
    version: str = "1.0"
    description: str = ""
    prompt_template: str = ""
    required_tools: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# LLM Response Types
# ══════════════════════════════════════════════════════════════════════


class LLMUsage(BaseModel):
    """Token usage from an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0


class LLMStreamChunk(BaseModel):
    """A single chunk from a streaming LLM response."""

    type: (
        str  # "text_delta", "tool_call_start", "tool_call_delta", "tool_call_end", "usage", "stop"
    )
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None
    usage: LLMUsage | None = None
    stop_reason: StopReason | None = None


class LLMResponse(BaseModel):
    """Complete (non-streaming) LLM response."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str = ""
    latency_ms: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Sub-Agent Result
# ══════════════════════════════════════════════════════════════════════


class SubAgentResult(BaseModel):
    """Result returned by a sub-agent to its parent."""

    child_session_id: str
    output: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    status: ToolStatus = ToolStatus.SUCCESS
    error: str | None = None
    turns_used: int = 0
    tokens_used: LLMUsage = Field(default_factory=LLMUsage)


# ══════════════════════════════════════════════════════════════════════
# Execution Context (passed to tools, middleware, hooks)
# ══════════════════════════════════════════════════════════════════════


class ExecutionContext(BaseModel):
    """
    Runtime context available to tools, middleware, and hooks.
    Carries session info, agent definition, auth context, and references to services.

    Auth fields (user_id, tenant_id) are pass-throughs from your web app.
    The framework never handles auth — your app authenticates, then passes
    the identity here so tools, middleware, and persistence are scoped correctly.
    """

    session_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    turn_number: int = 0
    parent_session_id: str | None = None

    # Auth context — passed through from your web app's auth layer
    user_id: str = ""
    tenant_id: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)

    working_memory: Any | None = Field(default=None, exclude=True)
    repository: Any | None = Field(default=None, exclude=True)
    active_skills: list[Any] = Field(default_factory=list, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def add_skill(self, skill: Any) -> None:
        """Add a skill mid-session (injected into prompt on next turn)."""
        self.active_skills.append(skill)

    def remove_skill(self, skill_name: str) -> None:
        """Remove a skill mid-session by name."""
        self.active_skills = [s for s in self.active_skills if getattr(s, "name", "") != skill_name]
