"""
Corza Agent Framework — API Schemas

Pydantic models for API request/response payloads.
"""
from typing import Any

from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════════════
# Session Endpoints
# ══════════════════════════════════════════════════════════════════════

class CreateSessionRequest(BaseModel):
    agent_id: str = Field(..., description="Agent definition ID or name")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata (tenant_id, case_id, etc.)")


class CreateSessionResponse(BaseModel):
    session_id: str
    agent_id: str
    status: str
    created_at: str


class SessionResponse(BaseModel):
    id: str
    agent_id: str
    status: str
    parent_session_id: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# Message Endpoints
# ══════════════════════════════════════════════════════════════════════

class SendMessageRequest(BaseModel):
    content: str = Field(..., description="User message content")
    variables: dict[str, Any] = Field(default_factory=dict, description="Template variables for prompt rendering")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    stream: bool = Field(default=True, description="Whether to stream the response via SSE")


class SendMessageResponse(BaseModel):
    session_id: str
    status: str
    message: str = ""


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: Any
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    model: str | None = None
    token_count: int = 0
    created_at: str


# ══════════════════════════════════════════════════════════════════════
# Tool Endpoints
# ══════════════════════════════════════════════════════════════════════

class ToolResponse(BaseModel):
    name: str
    description: str
    tool_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    permission_level: str = "auto_approve"
    tags: list[str] = Field(default_factory=list)


class RegisterToolRequest(BaseModel):
    name: str
    description: str
    tool_type: str = "function"
    parameters: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    permission_level: str = "auto_approve"
    timeout_seconds: int = 30
    tags: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Agent Definition Endpoints
# ══════════════════════════════════════════════════════════════════════

class AgentDefinitionResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    model: str
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    max_turns: int = 50
    temperature: float = 0.0


class RegisterAgentRequest(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    max_turns: int = 50
    max_tokens_per_turn: int = 8192
    temperature: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# Artifacts
# ══════════════════════════════════════════════════════════════════════

class ArtifactResponse(BaseModel):
    id: str
    artifact_type: str
    name: str
    content: str | None = None
    content_json: dict | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


# ══════════════════════════════════════════════════════════════════════
# Audit
# ══════════════════════════════════════════════════════════════════════

class AuditLogResponse(BaseModel):
    entries: list[dict[str, Any]]
    count: int


# ══════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.3.0"
    tools_count: int = 0
    agents_count: int = 0
    middleware: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Structured error response with correlation ID for log tracing."""
    error_id: str = Field(description="Unique error ID — include in bug reports")
    error: str = Field(description="Human-readable error message")
    error_type: str = Field(default="server_error", description="Error category")
    status_code: int = 500
    recoverable: bool = False
