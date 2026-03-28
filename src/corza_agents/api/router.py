"""
Corza Agent Framework — FastAPI Router

Thin HTTP adapter over the AgentService. All business logic lives
in AgentService — this module only maps HTTP ↔ service calls.

Features:
- Structured error responses with correlation IDs
- SSE streaming with heartbeat and Last-Event-ID reconnection
- Pagination on message history
- Session resume from WAITING_INPUT / FAILED states

Usage:
    from corza_agents.api import create_agent_router

    router = create_agent_router(orchestrator, agent_definitions)
    app.include_router(router, prefix="/api/agent")
"""

import uuid

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from corza_agents.api.schemas import (
    AgentDefinitionResponse,
    ArtifactResponse,
    AuditLogResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    RegisterAgentRequest,
    RegisterToolRequest,
    SendMessageRequest,
    SendMessageResponse,
    SessionResponse,
    ToolResponse,
)
from corza_agents.api.service import AgentService
from corza_agents.core.types import AgentDefinition, SessionStatus
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.streaming.sse import sse_response

log = structlog.get_logger("corza_agents.api")


def _error_response(
    status_code: int,
    message: str,
    error_type: str = "server_error",
    recoverable: bool = False,
) -> JSONResponse:
    """Build a structured error response with a unique error_id."""
    error_id = str(uuid.uuid4())[:12]
    log.error(
        "api_error",
        error_id=error_id,
        status=status_code,
        error_type=error_type,
        message=message[:500],
    )
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_id=error_id,
            error=message,
            error_type=error_type,
            status_code=status_code,
            recoverable=recoverable,
        ).model_dump(),
    )


def create_agent_router(
    orchestrator: Orchestrator,
    agent_definitions: dict[str, AgentDefinition] | None = None,
    service: AgentService | None = None,
    admin_only: bool = True,
) -> APIRouter:
    """
    Create a FastAPI router for the agent framework.

    Args:
        orchestrator: The configured Orchestrator instance.
        agent_definitions: Named agent definitions (brain, analyst, etc.).
        service: Optional pre-configured AgentService. If None, one is
                 created from orchestrator + agent_definitions.
        admin_only: If True (default), runtime tool/agent registration
                    endpoints (POST /tools, POST /agents) return 403.
                    Set False to enable runtime registration.

    Returns:
        APIRouter to mount on your FastAPI app.
    """
    if service is None:
        service = AgentService(orchestrator, agent_definitions)

    router = APIRouter(tags=["Agent Framework"])

    # ══════════════════════════════════════════════════════════════
    # Health
    # ══════════════════════════════════════════════════════════════

    @router.get("/health", response_model=HealthResponse)
    async def health():
        data = service.health()
        return HealthResponse(**data)

    # ══════════════════════════════════════════════════════════════
    # Sessions
    # ══════════════════════════════════════════════════════════════

    @router.post("/sessions", response_model=CreateSessionResponse)
    async def create_session(req: CreateSessionRequest):
        try:
            session = await service.create_session(req.agent_id, metadata=req.metadata)
        except KeyError as e:
            return _error_response(404, str(e), "not_found")
        return CreateSessionResponse(
            session_id=session.id,
            agent_id=session.agent_id,
            status="idle",
            created_at=session.created_at.isoformat(),
        )

    @router.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        session = await service.get_session(session_id)
        if not session:
            return _error_response(404, f"Session {session_id} not found", "not_found")
        return SessionResponse(
            id=session.id,
            agent_id=session.agent_id,
            status=session.status.value,
            parent_session_id=session.parent_session_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            completed_at=session.completed_at.isoformat() if session.completed_at else None,
            turn_count=session.turn_count,
            total_input_tokens=session.total_input_tokens,
            total_output_tokens=session.total_output_tokens,
            total_cost_usd=session.total_cost_usd,
            error=session.error,
            metadata=session.metadata,
        )

    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = await service.get_session(session_id)
        if not session:
            return _error_response(404, f"Session {session_id} not found", "not_found")
        await service.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}

    # ══════════════════════════════════════════════════════════════
    # Session Cancel — nuclear stop
    # ══════════════════════════════════════════════════════════════

    @router.post("/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str):
        """
        Nuclear stop — cancel a running session and all its sub-agent sessions.

        Sets the session (and all children) to CANCELLED status. Running
        agent loops detect this at the top of the next turn and exit.
        """
        session = await service.get_session(session_id)
        if not session:
            return _error_response(404, f"Session {session_id} not found", "not_found")
        count = await orchestrator.cancel(session_id)
        return {
            "status": "cancelled",
            "session_id": session_id,
            "sessions_cancelled": count,
        }

    # ══════════════════════════════════════════════════════════════
    # Session Resume — retry from WAITING_INPUT or FAILED
    # ══════════════════════════════════════════════════════════════

    @router.post("/sessions/{session_id}/resume")
    async def resume_session(session_id: str, req: SendMessageRequest, request: Request):
        """
        Resume a session stuck in WAITING_INPUT or FAILED state.
        Resets status to IDLE, then sends the message normally.
        """
        session = await service.get_session(session_id)
        if not session:
            return _error_response(404, f"Session {session_id} not found", "not_found")

        if session.status not in (SessionStatus.WAITING_INPUT, SessionStatus.FAILED):
            return _error_response(
                409,
                f"Session is {session.status.value}, not resumable. "
                f"Only WAITING_INPUT or FAILED sessions can be resumed.",
                "conflict",
            )

        # Reset session to allow new messages
        await orchestrator.repo.update_session(
            session_id,
            status=SessionStatus.IDLE,
            error=None,
        )

        if req.stream:
            events = service.send_message(
                session_id,
                req.content,
                req.metadata,
                req.variables,
            )
            return await sse_response(request, events)
        else:
            final_output, final_error = await service.send_message_sync(
                session_id,
                req.content,
                req.metadata,
                req.variables,
            )
            if final_error:
                return _error_response(500, final_error, "agent_error", recoverable=True)
            return SendMessageResponse(
                session_id=session_id,
                status="completed",
                message=final_output,
            )

    # ══════════════════════════════════════════════════════════════
    # Messages — Send + Stream
    # ══════════════════════════════════════════════════════════════

    @router.post("/sessions/{session_id}/messages")
    async def send_message(session_id: str, req: SendMessageRequest, request: Request):
        """
        Send a message to an agent session.
        If stream=True (default), returns SSE stream with heartbeat.
        If stream=False, returns JSON with final result.
        """
        try:
            if req.stream:
                events = service.send_message(
                    session_id,
                    req.content,
                    req.metadata,
                    req.variables,
                )
                return await sse_response(request, events)
            else:
                final_output, final_error = await service.send_message_sync(
                    session_id,
                    req.content,
                    req.metadata,
                    req.variables,
                )
                if final_error:
                    return _error_response(500, final_error, "agent_error", recoverable=True)
                return SendMessageResponse(
                    session_id=session_id,
                    status="completed",
                    message=final_output,
                )
        except KeyError as e:
            return _error_response(404, str(e), "not_found")
        except Exception as e:
            return _error_response(500, str(e)[:500], "server_error")

    @router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
    async def get_messages(
        session_id: str,
        include_summarized: bool = False,
        limit: int = Query(default=100, ge=1, le=1000, description="Max messages to return"),
        offset: int = Query(default=0, ge=0, description="Skip first N messages"),
    ):
        """Get message history with pagination."""
        messages = await service.get_messages(session_id, include_summarized)
        paginated = messages[offset : offset + limit]
        return [
            MessageResponse(
                id=m.id,
                session_id=m.session_id,
                role=m.role.value,
                content=m.content,
                tool_calls=[tc.model_dump() for tc in m.tool_calls] if m.tool_calls else None,
                tool_call_id=m.tool_call_id,
                tool_name=m.tool_name,
                model=m.model,
                token_count=m.token_count,
                created_at=m.created_at.isoformat(),
            )
            for m in paginated
        ]

    # ══════════════════════════════════════════════════════════════
    # Artifacts
    # ══════════════════════════════════════════════════════════════

    @router.get("/sessions/{session_id}/artifacts", response_model=list[ArtifactResponse])
    async def get_artifacts(session_id: str, artifact_type: str | None = None):
        artifacts = await service.get_artifacts(session_id, artifact_type)
        return [
            ArtifactResponse(
                id=a["id"],
                artifact_type=a["type"],
                name=a["name"],
                content=a.get("content"),
                content_json=a.get("content_json"),
                metadata=a.get("metadata", {}),
                created_at=a.get("created_at", ""),
            )
            for a in artifacts
        ]

    # ══════════════════════════════════════════════════════════════
    # Audit
    # ══════════════════════════════════════════════════════════════

    @router.get("/sessions/{session_id}/audit")
    async def get_audit_log(session_id: str):
        entries = await service.get_audit_log(session_id)
        return AuditLogResponse(entries=entries, count=len(entries))

    # ══════════════════════════════════════════════════════════════
    # Tools
    # ══════════════════════════════════════════════════════════════

    @router.get("/tools", response_model=list[ToolResponse])
    async def list_tools():
        return [
            ToolResponse(
                name=t.name,
                description=t.description,
                tool_type=t.tool_type.value,
                parameters=t.json_schema or {},
                permission_level=t.permission_level,
                tags=t.tags,
            )
            for t in service.list_tools()
        ]

    @router.post("/tools")
    async def register_tool(req: RegisterToolRequest):
        if admin_only:
            return _error_response(
                403,
                "Runtime tool registration is disabled. "
                "Set admin_only=False in create_agent_router() to enable, "
                "or register tools programmatically before starting the server.",
                "forbidden",
            )
        service.register_tool(
            name=req.name,
            description=req.description,
            tool_type=req.tool_type,
            parameters=req.parameters,
            config=req.config,
            permission_level=req.permission_level,
            timeout_seconds=req.timeout_seconds,
            tags=req.tags,
        )
        return {"status": "registered", "name": req.name}

    # ══════════════════════════════════════════════════════════════
    # Agent Definitions
    # ══════════════════════════════════════════════════════════════

    @router.get("/agents", response_model=list[AgentDefinitionResponse])
    async def list_agents():
        return [
            AgentDefinitionResponse(
                id=a.id,
                name=a.name,
                description=a.description,
                model=a.model,
                tools=a.tools,
                skills=a.skills,
                max_turns=a.max_turns,
                temperature=a.temperature,
            )
            for a in service.list_agents()
        ]

    @router.post("/agents")
    async def register_agent(req: RegisterAgentRequest):
        if admin_only:
            return _error_response(
                403,
                "Runtime agent registration is disabled. "
                "Set admin_only=False in create_agent_router() to enable, "
                "or register agents programmatically before starting the server.",
                "forbidden",
            )
        agent_def = AgentDefinition(
            name=req.name,
            description=req.description,
            system_prompt=req.system_prompt,
            model=req.model,
            tools=req.tools,
            skills=req.skills,
            max_turns=req.max_turns,
            max_tokens_per_turn=req.max_tokens_per_turn,
            temperature=req.temperature,
            metadata=req.metadata,
        )
        service.register_agent(req.name, agent_def)
        return {"status": "registered", "name": req.name, "id": agent_def.id}

    # ══════════════════════════════════════════════════════════════
    # Memory
    # ══════════════════════════════════════════════════════════════

    @router.get("/memory/{agent_id}")
    async def get_agent_memory(agent_id: str, memory_type: str | None = None):
        memories = await service.get_agent_memories(agent_id, memory_type)
        return {"agent_id": agent_id, "memories": memories, "count": len(memories)}

    @router.put("/memory/{agent_id}/{key}")
    async def set_agent_memory(agent_id: str, key: str, request: Request):
        body = await request.json()
        value = body.get("value")
        memory_type = body.get("type", "long_term")
        await service.set_agent_memory(agent_id, key, value, memory_type)
        return {"status": "ok", "agent_id": agent_id, "key": key}

    return router
