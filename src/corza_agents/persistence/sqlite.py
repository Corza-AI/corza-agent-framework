"""
Corza Agent Framework — SQLite Repository

Lightweight local persistence using aiosqlite.
Requires: pip install "corza-agents[sqlite]"

Data persists to a local .db file. No server needed.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from corza_agents.core.types import (
    AgentMessage,
    AgentSession,
    MessageRole,
    SessionStatus,
    ToolCall,
)
from corza_agents.persistence.base import BaseRepository

log = structlog.get_logger("corza_agents.persistence.sqlite")


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS af_sessions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    tenant_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    status TEXT NOT NULL DEFAULT 'idle',
    config TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    turn_count INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS ix_af_sessions_user ON af_sessions(user_id);
CREATE INDEX IF NOT EXISTS ix_af_sessions_tenant_user ON af_sessions(tenant_id, user_id, status);

CREATE TABLE IF NOT EXISTS af_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    model TEXT,
    created_at TEXT NOT NULL,
    is_summarized INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_af_messages_session ON af_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS af_tool_executions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_id TEXT,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    input TEXT NOT NULL DEFAULT '{}',
    output TEXT,
    status TEXT NOT NULL DEFAULT 'success',
    duration_ms REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS af_artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    name TEXT NOT NULL,
    content TEXT,
    content_json TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS af_audit_log (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    detail TEXT,
    llm_model TEXT,
    llm_input_tokens INTEGER,
    llm_output_tokens INTEGER,
    llm_latency_ms REAL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS af_memory (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'long_term',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_af_memory_agent_key ON af_memory(agent_id, key);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


class SQLiteRepository(BaseRepository):
    """
    SQLite persistence backend using aiosqlite.

    Usage:
        repo = SQLiteRepository("agents.db")
        await repo.initialize()
    """

    def __init__(self, db_path: str = "corza_agents.db"):
        self._db_path = db_path
        self._db = None

    async def initialize(self) -> None:
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite is required for SQLiteRepository. "
                "Install it with: pip install 'corza-agents[sqlite]'"
            )
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(CREATE_TABLES_SQL)
        await self._db.commit()
        log.info("sqlite_repository_initialized", db_path=self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _ensure_db(self):
        if not self._db:
            raise RuntimeError("SQLiteRepository not initialized. Call initialize() first.")
        return self._db

    # ── Sessions ──────────────────────────────────────────────────

    async def create_session(self, session: AgentSession) -> AgentSession:
        db = self._ensure_db()
        await db.execute(
            """INSERT INTO af_sessions
               (id, agent_id, user_id, tenant_id, parent_session_id, status, config, metadata,
                created_at, updated_at, total_input_tokens, total_output_tokens,
                total_cost_usd, turn_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session.id,
                session.agent_id,
                session.user_id,
                session.tenant_id,
                session.parent_session_id,
                session.status.value,
                _json_dumps(session.config),
                _json_dumps(session.metadata),
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.total_input_tokens,
                session.total_output_tokens,
                session.total_cost_usd,
                session.turn_count,
            ),
        )
        await db.commit()
        return session

    async def get_session(self, session_id: str) -> AgentSession | None:
        db = self._ensure_db()
        cursor = await db.execute("SELECT * FROM af_sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._session_from_row(row)

    async def update_session(self, session_id: str, **kwargs) -> None:
        db = self._ensure_db()
        updates = {k: v for k, v in kwargs.items() if v is not None}
        if "status" in updates and isinstance(updates["status"], SessionStatus):
            updates["status"] = updates["status"].value
        if "config" in updates and isinstance(updates["config"], dict):
            updates["config"] = _json_dumps(updates["config"])
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = _json_dumps(updates["metadata"])
        if "completed_at" in updates and isinstance(updates["completed_at"], datetime):
            updates["completed_at"] = updates["completed_at"].isoformat()
        updates["updated_at"] = _now_iso()
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        await db.execute(f"UPDATE af_sessions SET {set_clause} WHERE id = ?", values)
        await db.commit()

    async def get_sessions_for_user(
        self,
        user_id: str,
        tenant_id: str = "",
        status: str | None = None,
        limit: int = 50,
    ) -> list:
        db = self._ensure_db()
        query = "SELECT * FROM af_sessions WHERE user_id = ?"
        params: list = [user_id]
        if tenant_id:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._session_from_row(r) for r in rows]

    async def delete_session(self, session_id: str) -> None:
        db = self._ensure_db()
        for table in ("af_tool_executions", "af_messages", "af_artifacts", "af_audit_log"):
            await db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM af_sessions WHERE id = ?", (session_id,))
        await db.commit()

    # ── Messages ──────────────────────────────────────────────────

    async def add_message(self, message: AgentMessage) -> AgentMessage:
        db = self._ensure_db()
        content = message.content
        if isinstance(content, list):
            content = _json_dumps(content)
        tool_calls_data = None
        if message.tool_calls:
            tool_calls_data = _json_dumps([tc.model_dump() for tc in message.tool_calls])
        await db.execute(
            """INSERT INTO af_messages
               (id, session_id, role, content, tool_calls, tool_call_id,
                tool_name, token_count, model, created_at, is_summarized)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                message.session_id,
                message.role.value,
                content,
                tool_calls_data,
                message.tool_call_id,
                message.tool_name,
                message.token_count,
                message.model,
                message.created_at.isoformat(),
                int(message.is_summarized),
            ),
        )
        await db.commit()
        return message

    async def get_messages(
        self,
        session_id: str,
        include_summarized: bool = False,
    ) -> list[AgentMessage]:
        db = self._ensure_db()
        if include_summarized:
            cursor = await db.execute(
                "SELECT * FROM af_messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM af_messages WHERE session_id = ? AND is_summarized = 0 ORDER BY created_at",
                (session_id,),
            )
        rows = await cursor.fetchall()
        return [self._message_from_row(r) for r in rows]

    async def mark_messages_summarized(
        self,
        session_id: str,
        message_ids: list[str],
    ) -> None:
        if not message_ids:
            return
        db = self._ensure_db()
        placeholders = ",".join("?" for _ in message_ids)
        await db.execute(
            f"UPDATE af_messages SET is_summarized = 1 WHERE session_id = ? AND id IN ({placeholders})",
            [session_id] + message_ids,
        )
        await db.commit()

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
        db = self._ensure_db()
        output_json = output_data
        if isinstance(output_data, str):
            output_json = output_data
        else:
            output_json = _json_dumps(output_data)
        await db.execute(
            """INSERT INTO af_tool_executions
               (id, session_id, message_id, tool_call_id, tool_name,
                input, output, status, duration_ms, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _uuid(),
                session_id,
                message_id,
                tool_call_id,
                tool_name,
                _json_dumps(input_data),
                output_json,
                status,
                duration_ms,
                error,
                _now_iso(),
            ),
        )
        await db.commit()

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
        db = self._ensure_db()
        artifact_id = _uuid()
        await db.execute(
            """INSERT INTO af_artifacts
               (id, session_id, artifact_type, name, content, content_json, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                session_id,
                artifact_type,
                name,
                content,
                _json_dumps(content_json) if content_json else None,
                _json_dumps(metadata or {}),
                _now_iso(),
            ),
        )
        await db.commit()
        return artifact_id

    async def get_artifacts(
        self,
        session_id: str,
        artifact_type: str | None = None,
    ) -> list[dict]:
        db = self._ensure_db()
        if artifact_type:
            cursor = await db.execute(
                "SELECT * FROM af_artifacts WHERE session_id = ? AND artifact_type = ? ORDER BY created_at",
                (session_id, artifact_type),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM af_artifacts WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "type": r["artifact_type"],
                "name": r["name"],
                "content": r["content"],
                "content_json": json.loads(r["content_json"]) if r["content_json"] else None,
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
            }
            for r in rows
        ]

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
        db = self._ensure_db()
        await db.execute(
            """INSERT INTO af_audit_log
               (id, session_id, event_type, actor, action, detail,
                llm_model, llm_input_tokens, llm_output_tokens, llm_latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _uuid(),
                session_id,
                event_type,
                actor,
                action,
                _json_dumps(detail) if detail else None,
                llm_model,
                llm_input_tokens,
                llm_output_tokens,
                llm_latency_ms,
                _now_iso(),
            ),
        )
        await db.commit()

    async def get_audit_log(self, session_id: str) -> list[dict]:
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT * FROM af_audit_log WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "actor": r["actor"],
                "action": r["action"],
                "detail": json.loads(r["detail"]) if r["detail"] else None,
                "llm_model": r["llm_model"],
                "llm_input_tokens": r["llm_input_tokens"],
                "llm_output_tokens": r["llm_output_tokens"],
                "llm_latency_ms": r["llm_latency_ms"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    # ── Memory Store ──────────────────────────────────────────────

    async def get_memory(self, agent_id: str, key: str) -> Any | None:
        db = self._ensure_db()
        cursor = await db.execute(
            "SELECT value FROM af_memory WHERE agent_id = ? AND key = ?",
            (agent_id, key),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return json.loads(row["value"])

    async def set_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "long_term",
        session_id: str | None = None,
    ) -> None:
        db = self._ensure_db()
        now = _now_iso()
        value_json = _json_dumps(value)
        # Upsert via INSERT OR REPLACE
        cursor = await db.execute(
            "SELECT id, created_at FROM af_memory WHERE agent_id = ? AND key = ?",
            (agent_id, key),
        )
        existing = await cursor.fetchone()
        if existing:
            await db.execute(
                "UPDATE af_memory SET value = ?, memory_type = ?, session_id = ?, updated_at = ? WHERE id = ?",
                (value_json, memory_type, session_id, now, existing["id"]),
            )
        else:
            await db.execute(
                """INSERT INTO af_memory
                   (id, agent_id, memory_type, key, value, session_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (_uuid(), agent_id, memory_type, key, value_json, session_id, now, now),
            )
        await db.commit()

    async def list_memories(
        self,
        agent_id: str,
        memory_type: str | None = None,
    ) -> list[dict]:
        db = self._ensure_db()
        if memory_type:
            cursor = await db.execute(
                "SELECT * FROM af_memory WHERE agent_id = ? AND memory_type = ? ORDER BY updated_at DESC",
                (agent_id, memory_type),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM af_memory WHERE agent_id = ? ORDER BY updated_at DESC",
                (agent_id,),
            )
        rows = await cursor.fetchall()
        return [
            {
                "key": r["key"],
                "value": json.loads(r["value"]),
                "type": r["memory_type"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    # ── Row converters ────────────────────────────────────────────

    @staticmethod
    def _session_from_row(row) -> AgentSession:
        return AgentSession(
            id=row["id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"] if "user_id" in row.keys() else "",
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else "",
            parent_session_id=row["parent_session_id"],
            status=SessionStatus(row["status"]),
            config=json.loads(row["config"]) if row["config"] else {},
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            total_cost_usd=row["total_cost_usd"],
            turn_count=row["turn_count"],
            error=row["error"],
        )

    @staticmethod
    def _message_from_row(row) -> AgentMessage:
        content = row["content"]
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                content = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        tool_calls = None
        if row["tool_calls"]:
            tool_calls = [ToolCall(**tc) for tc in json.loads(row["tool_calls"])]
        return AgentMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=MessageRole(row["role"]),
            content=content,
            tool_calls=tool_calls,
            tool_call_id=row["tool_call_id"],
            tool_name=row["tool_name"],
            token_count=row["token_count"],
            model=row["model"],
            created_at=datetime.fromisoformat(row["created_at"]),
            is_summarized=bool(row["is_summarized"]),
        )
