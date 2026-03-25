"""
Corza Agent Framework — In-Memory Repository

Zero-dependency persistence backend using plain Python dicts.
Perfect for testing, prototyping, and single-run scripts.
Data is lost when the process exits.
"""
import copy
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from corza_agents.core.types import AgentMessage, AgentSession, SessionStatus
from corza_agents.persistence.base import BaseRepository

log = structlog.get_logger("corza_agents.persistence.memory")


class InMemoryRepository(BaseRepository):
    """
    In-memory persistence backend. No database required.

    All data lives in Python dicts and is lost on process exit.
    Thread-safe for single-event-loop async usage.
    """

    def __init__(self):
        self._sessions: dict[str, AgentSession] = {}
        self._messages: dict[str, list[AgentMessage]] = {}
        self._tool_executions: list[dict] = []
        self._artifacts: dict[str, list[dict]] = {}
        self._audit_log: list[dict] = []
        self._memory: dict[str, dict[str, dict]] = {}

    async def initialize(self) -> None:
        log.debug("in_memory_repository_initialized")

    async def close(self) -> None:
        pass

    # ── Sessions ──────────────────────────────────────────────────

    async def create_session(self, session: AgentSession) -> AgentSession:
        self._sessions[session.id] = session.model_copy()
        self._messages.setdefault(session.id, [])
        self._artifacts.setdefault(session.id, [])
        return session

    async def get_session(self, session_id: str) -> AgentSession | None:
        s = self._sessions.get(session_id)
        return s.model_copy() if s else None

    async def update_session(self, session_id: str, **kwargs) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        updates = {k: v for k, v in kwargs.items() if v is not None}
        if "status" in updates and isinstance(updates["status"], SessionStatus):
            pass  # keep as enum, pydantic handles it
        updates["updated_at"] = datetime.now(UTC)
        self._sessions[session_id] = session.model_copy(update=updates)

    async def get_sessions_for_user(
        self, user_id: str, tenant_id: str = "",
        status: str | None = None, limit: int = 50,
    ) -> list:
        sessions = [
            s for s in self._sessions.values()
            if s.user_id == user_id and (not tenant_id or s.tenant_id == tenant_id)
        ]
        if status:
            sessions = [s for s in sessions if s.status.value == status]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return [s.model_copy() for s in sessions[:limit]]

    async def get_child_sessions(self, parent_session_id: str) -> list:
        return [
            s.model_copy() for s in self._sessions.values()
            if s.parent_session_id == parent_session_id
        ]

    async def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)
        self._artifacts.pop(session_id, None)
        self._tool_executions[:] = [
            t for t in self._tool_executions if t.get("session_id") != session_id
        ]
        self._audit_log[:] = [
            a for a in self._audit_log if a.get("session_id") != session_id
        ]

    async def purge_old_sessions(
        self, max_age_hours: int = 168, statuses: list[str] | None = None,
    ) -> int:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        to_delete = []
        for sid, s in self._sessions.items():
            if s.updated_at < cutoff:
                if statuses is None or s.status.value in statuses:
                    to_delete.append(sid)
        for sid in to_delete:
            await self.delete_session(sid)
        return len(to_delete)

    # ── Messages ──────────────────────────────────────────────────

    async def add_message(self, message: AgentMessage) -> AgentMessage:
        self._messages.setdefault(message.session_id, []).append(message.model_copy())
        return message

    async def get_messages(
        self, session_id: str, include_summarized: bool = False,
    ) -> list[AgentMessage]:
        msgs = self._messages.get(session_id, [])
        if not include_summarized:
            msgs = [m for m in msgs if not m.is_summarized]
        return [m.model_copy() for m in msgs]

    async def mark_messages_summarized(
        self, session_id: str, message_ids: list[str],
    ) -> None:
        ids_set = set(message_ids)
        msgs = self._messages.get(session_id, [])
        for i, m in enumerate(msgs):
            if m.id in ids_set:
                msgs[i] = m.model_copy(update={"is_summarized": True})

    # ── Tool Executions ───────────────────────────────────────────

    async def log_tool_execution(
        self,
        session_id: str,
        message_id: str | None,
        tool_call_id: str,
        tool_name: str,
        input_data: dict,
        output_data: Any,
        status: str,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        self._tool_executions.append({
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "message_id": message_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "input": input_data,
            "output": output_data,
            "status": status,
            "duration_ms": duration_ms,
            "error": error,
            "created_at": datetime.now(UTC).isoformat(),
        })

    # ── Artifacts ─────────────────────────────────────────────────

    async def save_artifact(
        self,
        session_id: str,
        artifact_type: str,
        name: str,
        content: str | None = None,
        content_json: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        artifact_id = str(uuid.uuid4())
        self._artifacts.setdefault(session_id, []).append({
            "id": artifact_id,
            "session_id": session_id,
            "type": artifact_type,
            "name": name,
            "content": content,
            "content_json": content_json,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        })
        return artifact_id

    async def get_artifacts(
        self, session_id: str, artifact_type: str | None = None,
    ) -> list[dict]:
        artifacts = self._artifacts.get(session_id, [])
        if artifact_type:
            artifacts = [a for a in artifacts if a["type"] == artifact_type]
        return copy.deepcopy(artifacts)

    # ── Audit Log ─────────────────────────────────────────────────

    async def log_audit(
        self,
        session_id: str,
        event_type: str,
        action: str,
        actor: str = "",
        detail: dict | None = None,
        llm_model: str | None = None,
        llm_input_tokens: int | None = None,
        llm_output_tokens: int | None = None,
        llm_latency_ms: float | None = None,
    ) -> None:
        self._audit_log.append({
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "detail": detail,
            "llm_model": llm_model,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
            "llm_latency_ms": llm_latency_ms,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    async def get_audit_log(self, session_id: str) -> list[dict]:
        return [
            e for e in self._audit_log if e["session_id"] == session_id
        ]

    # ── Memory Store ──────────────────────────────────────────────

    async def get_memory(self, agent_id: str, key: str) -> Any | None:
        agent_mem = self._memory.get(agent_id, {})
        entry = agent_mem.get(key)
        return entry["value"] if entry else None

    async def set_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "long_term",
        session_id: str | None = None,
    ) -> None:
        if agent_id not in self._memory:
            self._memory[agent_id] = {}
        now = datetime.now(UTC).isoformat()
        existing = self._memory[agent_id].get(key)
        if existing:
            existing["value"] = value
            existing["type"] = memory_type
            existing["updated_at"] = now
            if session_id:
                existing["session_id"] = session_id
        else:
            self._memory[agent_id][key] = {
                "value": value,
                "type": memory_type,
                "session_id": session_id,
                "updated_at": now,
            }

    async def list_memories(
        self, agent_id: str, memory_type: str | None = None,
    ) -> list[dict]:
        agent_mem = self._memory.get(agent_id, {})
        result = []
        for key, entry in agent_mem.items():
            if memory_type and entry.get("type") != memory_type:
                continue
            result.append({
                "key": key,
                "value": entry["value"],
                "type": entry.get("type", "long_term"),
                "updated_at": entry.get("updated_at"),
            })
        return sorted(result, key=lambda x: x.get("updated_at", ""), reverse=True)
