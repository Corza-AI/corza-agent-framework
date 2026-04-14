"""
Corza Agent Framework — Framework-Agnostic Service Layer

No web framework imports. Can be used by FastAPI, Flask, Django, CLI, etc.
All business logic lives here; HTTP routers are thin adapters.

Usage:
    from corza_agents.api.service import AgentService

    service = AgentService(orchestrator, {"brain": brain_def})
    session = await service.create_session("brain")
    async for event in service.send_message(session.id, "Hello"):
        print(event)
"""

from collections.abc import AsyncIterator
from typing import Any

import structlog

from corza_agents.core.types import (
    AgentDefinition,
    AgentMessage,
    AgentSession,
    RegisteredTool,
    ToolType,
    _uuid,
)
from corza_agents.api.run_registry import get_registry, subscribe_iter
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.streaming.events import StreamEvent

log = structlog.get_logger("corza_agents.api.service")


class AgentService:
    """
    Stateless service layer for agent operations.

    Delegates to Orchestrator and Repository via their public APIs.
    No HTTP, no framework — pure business logic.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        agent_definitions: dict[str, AgentDefinition] | None = None,
    ):
        self._orchestrator = orchestrator
        self._agents: dict[str, AgentDefinition] = dict(agent_definitions or {})

    # ── Agent Definitions ─────────────────────────────────────────

    def list_agents(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def get_agent(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def register_agent(self, name: str, agent_def: AgentDefinition) -> None:
        self._agents[name] = agent_def

    # ── Sessions ──────────────────────────────────────────────────

    async def create_session(
        self,
        agent_id: str,
        user_id: str = "",
        tenant_id: str = "",
        metadata: dict | None = None,
    ) -> AgentSession:
        if agent_id not in self._agents:
            raise KeyError(f"Agent '{agent_id}' not found. Available: {list(self._agents.keys())}")
        session_id = _uuid()
        session = AgentSession(
            id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )
        await self._orchestrator.repo.create_session(session)
        return session

    async def get_session(self, session_id: str) -> AgentSession | None:
        return await self._orchestrator.repo.get_session(session_id)

    async def get_sessions_for_user(
        self,
        user_id: str,
        tenant_id: str = "",
        status: str | None = None,
        limit: int = 50,
    ) -> list[AgentSession]:
        return await self._orchestrator.repo.get_sessions_for_user(
            user_id, tenant_id, status, limit
        )

    async def delete_session(self, session_id: str) -> None:
        await self._orchestrator.repo.delete_session(session_id)

    # ── Messages ──────────────────────────────────────────────────

    async def send_message(
        self,
        session_id: str,
        content: str,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream agent events for a message.

        The agent run is spawned as a detached task owned by the process-wide
        RunRegistry, NOT by the HTTP request. Subscribers (SSE consumers) can
        come and go — client disconnect does not cancel the underlying run.
        If a run is already in progress for this session, the caller
        subscribes to it and receives backlog + live events.
        """
        session = await self._orchestrator.repo.get_session(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")
        agent_def = self._agents.get(session.agent_id)
        if not agent_def:
            raise KeyError(f"Agent '{session.agent_id}' not found")

        registry = get_registry()

        def _factory() -> AsyncIterator[StreamEvent]:
            return self._orchestrator.run(
                session_id,
                content,
                agent_def,
                metadata,
                variables,
            )

        active, is_new = await registry.start(session_id, _factory)
        if not is_new:
            # A run was already in flight for this session. The new `content`
            # is NOT fed to the agent — we just attach to the existing run so
            # the caller sees its events. This matches the "one agent loop per
            # session" invariant of the framework. Log loudly so duplicate
            # POSTs (e.g. React strict-mode double-fires, client retries) are
            # visible in ops dashboards.
            log.warning(
                "duplicate_send_message_attached_to_existing_run",
                session_id=session_id,
                dropped_content_preview=(content[:120] if content else ""),
            )
        async for event in subscribe_iter(active):
            yield event

    async def subscribe_to_run(
        self,
        session_id: str,
    ) -> AsyncIterator[StreamEvent] | None:
        """
        Attach to an in-flight run for `session_id` without sending a message.
        Returns None if no run is active. Used by resume-stream endpoints.
        """
        registry = get_registry()
        active = await registry.get(session_id)
        if not active or active.done:
            return None
        return subscribe_iter(active)

    async def cancel_run(self, session_id: str) -> bool:
        """Cancel the in-flight run for this session, if any."""
        return await get_registry().cancel(session_id)

    async def send_message_sync(
        self,
        session_id: str,
        content: str,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> tuple[str, str | None]:
        """Non-streaming: returns (final_output, error_or_none)."""
        final_output = ""
        final_error = None
        async for event in self.send_message(session_id, content, metadata, variables):
            if event.type.value == "session.completed":
                final_output = event.data.get("final_output", "")
            elif event.type.value == "error":
                final_error = event.data.get("message", "Unknown error")
        return final_output, final_error

    async def get_messages(
        self,
        session_id: str,
        include_summarized: bool = False,
    ) -> list[AgentMessage]:
        return await self._orchestrator.repo.get_messages(session_id, include_summarized)

    # ── Artifacts ─────────────────────────────────────────────────

    async def get_artifacts(
        self,
        session_id: str,
        artifact_type: str | None = None,
    ) -> list[dict]:
        return await self._orchestrator.repo.get_artifacts(session_id, artifact_type)

    # ── Audit ─────────────────────────────────────────────────────

    async def get_audit_log(self, session_id: str) -> list[dict]:
        return await self._orchestrator.repo.get_audit_log(session_id)

    # ── Tools ─────────────────────────────────────────────────────

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._orchestrator.tools.tools.values())

    def register_tool(
        self,
        name: str,
        description: str,
        tool_type: str = "function",
        parameters: dict | None = None,
        config: dict | None = None,
        permission_level: str = "auto_approve",
        timeout_seconds: int = 30,
        tags: list[str] | None = None,
    ) -> None:
        tool = RegisteredTool(
            name=name,
            description=description,
            tool_type=ToolType(tool_type),
            json_schema=parameters or {},
            config=config or {},
            permission_level=permission_level,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )
        self._orchestrator.tools.register(tool)

    # ── Memory ────────────────────────────────────────────────────

    async def get_agent_memories(
        self,
        agent_id: str,
        memory_type: str | None = None,
    ) -> list[dict]:
        return await self._orchestrator.repo.list_memories(agent_id, memory_type)

    async def set_agent_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "long_term",
    ) -> None:
        await self._orchestrator.repo.set_memory(agent_id, key, value, memory_type)

    # ── Health ────────────────────────────────────────────────────

    def health(self) -> dict:
        return {
            "status": "ok",
            "tools_count": len(self._orchestrator.tools.tools),
            "agents_count": len(self._agents),
            "middleware": [m.name for m in self._orchestrator.middleware],
        }
