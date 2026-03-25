"""
Corza Agent Framework — PostgreSQL Data Access Layer

Async repository using SQLAlchemy + asyncpg.
Requires: pip install "corza-agents[postgres]"
"""
import json
import math
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from corza_agents.core.types import (
    AgentMessage,
    AgentSession,
    MessageRole,
    SessionStatus,
)
from corza_agents.persistence.base import BaseRepository
from corza_agents.persistence.models import (
    AgentArtifactModel,
    AgentAuditLogModel,
    AgentMemoryModel,
    AgentMessageModel,
    AgentSessionModel,
    AgentToolExecutionModel,
    Base,
)

log = structlog.get_logger("corza_agents.repository")


class PostgresRepository(BaseRepository):
    """
    PostgreSQL persistence backend using SQLAlchemy + asyncpg.

    All DB operations go through here. No other module touches
    SQLAlchemy models directly.
    """

    def __init__(
        self,
        db_url: str,
        echo: bool = False,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 1800,
    ):
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        self._engine = create_async_engine(
            db_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,
            echo=echo,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """
        Create tables if they don't exist and track schema version.

        On first run: creates all tables + records schema version.
        On subsequent runs: verifies schema version matches code.
        Logs a warning if versions mismatch (manual migration needed).
        """
        from corza_agents.persistence.models import SCHEMA_VERSION, AgentSchemaVersionModel

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Check / set schema version
        async with self._session_factory() as db:
            from sqlalchemy import select
            result = await db.execute(
                select(AgentSchemaVersionModel).where(AgentSchemaVersionModel.id == 1)
            )
            row = result.scalar_one_or_none()
            if row is None:
                db.add(AgentSchemaVersionModel(id=1, version=SCHEMA_VERSION))
                await db.commit()
                log.info("schema_initialized", version=SCHEMA_VERSION)
            elif row.version != SCHEMA_VERSION:
                log.warning(
                    "schema_version_mismatch",
                    db_version=row.version,
                    code_version=SCHEMA_VERSION,
                    message="Database schema is outdated. Run migrations or recreate tables.",
                )
            else:
                log.debug("schema_version_ok", version=SCHEMA_VERSION)

    async def close(self):
        await self._engine.dispose()

    def session(self) -> AsyncSession:
        return self._session_factory()

    # ══════════════════════════════════════════════════════════════════
    # Sessions
    # ══════════════════════════════════════════════════════════════════

    async def create_session(self, session: AgentSession) -> AgentSession:
        async with self.session() as db:
            model = AgentSessionModel(
                id=session.id,
                agent_id=session.agent_id,
                user_id=session.user_id,
                tenant_id=session.tenant_id,
                parent_session_id=session.parent_session_id,
                status=session.status.value,
                config=session.config,
                metadata_=session.metadata,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
            db.add(model)
            await db.commit()
            return session

    async def get_session(self, session_id: str) -> AgentSession | None:
        async with self.session() as db:
            result = await db.execute(
                select(AgentSessionModel).where(AgentSessionModel.id == session_id)
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            return self._session_from_model(model)

    async def update_session(self, session_id: str, **kwargs) -> None:
        async with self.session() as db:
            values = {k: v for k, v in kwargs.items() if v is not None}
            if "status" in values and isinstance(values["status"], SessionStatus):
                values["status"] = values["status"].value
            if "metadata" in values:
                values["metadata_"] = values.pop("metadata")
            values["updated_at"] = datetime.now(UTC)
            await db.execute(
                update(AgentSessionModel)
                .where(AgentSessionModel.id == session_id)
                .values(**values)
            )
            await db.commit()

    async def get_sessions_for_user(
        self, user_id: str, tenant_id: str = "",
        status: str | None = None, limit: int = 50,
    ) -> list["AgentSession"]:
        async with self.session() as db:
            query = select(AgentSessionModel).where(
                AgentSessionModel.user_id == user_id
            )
            if tenant_id:
                query = query.where(AgentSessionModel.tenant_id == tenant_id)
            if status:
                query = query.where(AgentSessionModel.status == status)
            query = query.order_by(AgentSessionModel.updated_at.desc()).limit(limit)
            result = await db.execute(query)
            return [self._session_from_model(m) for m in result.scalars().all()]

    async def delete_session(self, session_id: str) -> None:
        async with self.session() as db:
            result = await db.execute(
                select(AgentSessionModel).where(AgentSessionModel.id == session_id)
            )
            model = result.scalar_one_or_none()
            if model:
                await db.delete(model)
                await db.commit()

    # ══════════════════════════════════════════════════════════════════
    # Messages
    # ══════════════════════════════════════════════════════════════════

    async def add_message(self, message: AgentMessage) -> AgentMessage:
        async with self.session() as db:
            content = message.content
            if isinstance(content, list):
                content = json.loads(json.dumps(content, default=str))

            tool_calls_data = None
            if message.tool_calls:
                tool_calls_data = [tc.model_dump() for tc in message.tool_calls]

            model = AgentMessageModel(
                id=message.id,
                session_id=message.session_id,
                role=message.role.value,
                content=content,
                tool_calls=tool_calls_data,
                tool_call_id=message.tool_call_id,
                tool_name=message.tool_name,
                token_count=message.token_count,
                model=message.model,
                created_at=message.created_at,
                is_summarized=message.is_summarized,
            )
            db.add(model)
            await db.commit()
            return message

    async def get_messages(self, session_id: str,
                           include_summarized: bool = False) -> list[AgentMessage]:
        async with self.session() as db:
            query = (
                select(AgentMessageModel)
                .where(AgentMessageModel.session_id == session_id)
                .order_by(AgentMessageModel.created_at)
            )
            if not include_summarized:
                query = query.where(AgentMessageModel.is_summarized == False)  # noqa: E712
            result = await db.execute(query)
            return [self._message_from_model(m) for m in result.scalars().all()]

    async def mark_messages_summarized(self, session_id: str,
                                        message_ids: list[str]) -> None:
        """Mark messages as consumed by summarization."""
        if not message_ids:
            return
        async with self.session() as db:
            await db.execute(
                update(AgentMessageModel)
                .where(
                    and_(
                        AgentMessageModel.session_id == session_id,
                        AgentMessageModel.id.in_(message_ids),
                    )
                )
                .values(is_summarized=True)
            )
            await db.commit()

    # ══════════════════════════════════════════════════════════════════
    # Tool Executions
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _sanitize_json(obj: Any) -> Any:
        """Replace NaN/Infinity with None so PostgreSQL JSON accepts it."""
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: Repository._sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Repository._sanitize_json(v) for v in obj]
        return obj

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
        async with self.session() as db:
            output_json = output_data
            if isinstance(output_data, str):
                try:
                    output_json = json.loads(output_data)
                except (json.JSONDecodeError, TypeError):
                    output_json = {"text": output_data[:10000]}
            output_json = self._sanitize_json(output_json)
            input_data = self._sanitize_json(input_data)

            model = AgentToolExecutionModel(
                session_id=session_id,
                message_id=message_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                input=input_data,
                output=output_json,
                status=status,
                duration_ms=duration_ms,
                error=error,
            )
            db.add(model)
            await db.commit()

    # ══════════════════════════════════════════════════════════════════
    # Artifacts
    # ══════════════════════════════════════════════════════════════════

    async def save_artifact(
        self,
        session_id: str,
        artifact_type: str,
        name: str,
        content: str | None = None,
        content_json: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        async with self.session() as db:
            model = AgentArtifactModel(
                session_id=session_id,
                artifact_type=artifact_type,
                name=name,
                content=content,
                content_json=content_json,
                metadata_=metadata or {},
            )
            db.add(model)
            await db.commit()
            return model.id

    async def get_artifacts(self, session_id: str,
                            artifact_type: str | None = None) -> list[dict]:
        async with self.session() as db:
            query = select(AgentArtifactModel).where(
                AgentArtifactModel.session_id == session_id
            )
            if artifact_type:
                query = query.where(AgentArtifactModel.artifact_type == artifact_type)
            result = await db.execute(query.order_by(AgentArtifactModel.created_at))
            return [
                {
                    "id": m.id,
                    "type": m.artifact_type,
                    "name": m.name,
                    "content": m.content,
                    "content_json": m.content_json,
                    "metadata": m.metadata_,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in result.scalars().all()
            ]

    # ══════════════════════════════════════════════════════════════════
    # Audit Log
    # ══════════════════════════════════════════════════════════════════

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
        async with self.session() as db:
            model = AgentAuditLogModel(
                session_id=session_id,
                event_type=event_type,
                actor=actor,
                action=action,
                detail=detail,
                llm_model=llm_model,
                llm_input_tokens=llm_input_tokens,
                llm_output_tokens=llm_output_tokens,
                llm_latency_ms=llm_latency_ms,
            )
            db.add(model)
            await db.commit()

    # ══════════════════════════════════════════════════════════════════
    # Memory Store
    # ══════════════════════════════════════════════════════════════════

    async def get_memory(self, agent_id: str, key: str) -> Any | None:
        async with self.session() as db:
            result = await db.execute(
                select(AgentMemoryModel).where(
                    and_(
                        AgentMemoryModel.agent_id == agent_id,
                        AgentMemoryModel.key == key,
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            return model.value

    async def set_memory(self, agent_id: str, key: str, value: Any,
                         memory_type: str = "long_term",
                         session_id: str | None = None) -> None:
        async with self.session() as db:
            existing = await db.execute(
                select(AgentMemoryModel).where(
                    and_(
                        AgentMemoryModel.agent_id == agent_id,
                        AgentMemoryModel.key == key,
                    )
                )
            )
            model = existing.scalar_one_or_none()
            if model:
                model.value = value
                model.memory_type = memory_type
                model.updated_at = datetime.now(UTC)
                if session_id:
                    model.session_id = session_id
            else:
                model = AgentMemoryModel(
                    agent_id=agent_id,
                    memory_type=memory_type,
                    key=key,
                    value=value,
                    session_id=session_id,
                )
                db.add(model)
            await db.commit()

    async def list_memories(self, agent_id: str,
                            memory_type: str | None = None) -> list[dict]:
        async with self.session() as db:
            query = select(AgentMemoryModel).where(
                AgentMemoryModel.agent_id == agent_id
            )
            if memory_type:
                query = query.where(AgentMemoryModel.memory_type == memory_type)
            result = await db.execute(query.order_by(AgentMemoryModel.updated_at.desc()))
            return [
                {"key": m.key, "value": m.value, "type": m.memory_type,
                 "updated_at": m.updated_at.isoformat() if m.updated_at else None}
                for m in result.scalars().all()
            ]

    # ══════════════════════════════════════════════════════════════════
    # Audit Log Query
    # ══════════════════════════════════════════════════════════════════

    async def get_audit_log(self, session_id: str) -> list[dict]:
        async with self.session() as db:
            result = await db.execute(
                select(AgentAuditLogModel)
                .where(AgentAuditLogModel.session_id == session_id)
                .order_by(AgentAuditLogModel.timestamp)
            )
            return [
                {
                    "id": e.id,
                    "event_type": e.event_type,
                    "actor": e.actor,
                    "action": e.action,
                    "detail": e.detail,
                    "llm_model": e.llm_model,
                    "llm_input_tokens": e.llm_input_tokens,
                    "llm_output_tokens": e.llm_output_tokens,
                    "llm_latency_ms": e.llm_latency_ms,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                }
                for e in result.scalars().all()
            ]

    # ══════════════════════════════════════════════════════════════════
    # Model → Pydantic Converters
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _session_from_model(m: AgentSessionModel) -> AgentSession:
        return AgentSession(
            id=m.id,
            agent_id=m.agent_id,
            user_id=m.user_id or "",
            tenant_id=m.tenant_id or "",
            parent_session_id=m.parent_session_id,
            status=SessionStatus(m.status),
            config=m.config or {},
            metadata=m.metadata_ or {},
            created_at=m.created_at,
            updated_at=m.updated_at,
            completed_at=m.completed_at,
            total_input_tokens=m.total_input_tokens,
            total_output_tokens=m.total_output_tokens,
            total_cost_usd=m.total_cost_usd,
            turn_count=m.turn_count,
            error=m.error,
        )

    @staticmethod
    def _message_from_model(m: AgentMessageModel) -> AgentMessage:
        from corza_agents.core.types import ToolCall
        tool_calls = None
        if m.tool_calls:
            tool_calls = [ToolCall(**tc) for tc in m.tool_calls]
        return AgentMessage(
            id=m.id,
            session_id=m.session_id,
            role=MessageRole(m.role),
            content=m.content,
            tool_calls=tool_calls,
            tool_call_id=m.tool_call_id,
            tool_name=m.tool_name,
            token_count=m.token_count,
            model=m.model,
            created_at=m.created_at,
            is_summarized=m.is_summarized,
        )


# Backward compatibility alias
Repository = PostgresRepository
