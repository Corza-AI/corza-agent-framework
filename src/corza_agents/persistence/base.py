"""
Corza Agent Framework — Abstract Repository Interface

Defines the contract all persistence backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from corza_agents.core.types import AgentMessage, AgentSession


class BaseRepository(ABC):
    """
    Abstract base for all persistence backends.

    Implementations: PostgresRepository, InMemoryRepository,
    SQLiteRepository.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up storage (create tables, dirs, etc.). Call once on startup."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    # ══════════════════════════════════════════════════════════════════
    # Sessions
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def create_session(self, session: AgentSession) -> AgentSession: ...

    @abstractmethod
    async def get_session(self, session_id: str) -> AgentSession | None: ...

    @abstractmethod
    async def update_session(self, session_id: str, **kwargs) -> None: ...

    @abstractmethod
    async def get_sessions_for_user(
        self,
        user_id: str,
        tenant_id: str = "",
        status: str | None = None,
        limit: int = 50,
    ) -> list[AgentSession]: ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None: ...

    async def get_child_sessions(self, parent_session_id: str) -> list[AgentSession]:
        """Get all sessions spawned by a parent session."""
        return []

    async def purge_old_sessions(
        self,
        max_age_hours: int = 168,
        statuses: list[str] | None = None,
    ) -> int:
        """
        Delete sessions older than max_age_hours. Override in subclasses for
        efficient DB-level deletion. Default: no-op returning 0.
        """
        return 0

    # ══════════════════════════════════════════════════════════════════
    # Messages
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def add_message(self, message: AgentMessage) -> AgentMessage: ...

    @abstractmethod
    async def get_messages(
        self,
        session_id: str,
        include_summarized: bool = False,
    ) -> list[AgentMessage]: ...

    @abstractmethod
    async def mark_messages_summarized(
        self,
        session_id: str,
        message_ids: list[str],
    ) -> None: ...

    # ══════════════════════════════════════════════════════════════════
    # Tool Executions
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
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
    ) -> None: ...

    # ══════════════════════════════════════════════════════════════════
    # Artifacts
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def save_artifact(
        self,
        session_id: str,
        artifact_type: str,
        name: str,
        content: str | None = None,
        content_json: dict | None = None,
        metadata: dict | None = None,
    ) -> str: ...

    @abstractmethod
    async def get_artifacts(
        self,
        session_id: str,
        artifact_type: str | None = None,
    ) -> list[dict]: ...

    # ══════════════════════════════════════════════════════════════════
    # Audit Log
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
    async def get_audit_log(self, session_id: str) -> list[dict]: ...

    # ══════════════════════════════════════════════════════════════════
    # Memory Store
    # ══════════════════════════════════════════════════════════════════

    @abstractmethod
    async def get_memory(self, agent_id: str, key: str) -> Any | None: ...

    @abstractmethod
    async def set_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "long_term",
        session_id: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def list_memories(
        self,
        agent_id: str,
        memory_type: str | None = None,
    ) -> list[dict]: ...
