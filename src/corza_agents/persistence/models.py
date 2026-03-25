"""
Corza Agent Framework — SQLAlchemy Models

Async-first DB models for sessions, messages, tool executions, and artifacts.
These live alongside existing application tables — no conflicts.
"""
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for agent framework tables."""
    pass


def _utcnow():
    return datetime.now(UTC)


def _uuid():
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════
# Schema Version Tracking
# ══════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = 1  # Increment when schema changes

class AgentSchemaVersionModel(Base):
    __tablename__ = "af_schema_version"

    id = Column(Integer, primary_key=True, default=1)
    version = Column(Integer, nullable=False, default=SCHEMA_VERSION)
    applied_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ══════════════════════════════════════════════════════════════════════
# Sessions
# ══════════════════════════════════════════════════════════════════════

class AgentSessionModel(Base):
    __tablename__ = "af_sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    agent_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, default="", index=True)
    tenant_id = Column(String(255), nullable=False, default="", index=True)
    parent_session_id = Column(String(36), ForeignKey("af_sessions.id"), nullable=True, index=True)
    status = Column(String(20), nullable=False, default="idle", index=True)
    config = Column(JSON, nullable=False, default=dict)
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_input_tokens = Column(Integer, nullable=False, default=0)
    total_output_tokens = Column(Integer, nullable=False, default=0)
    total_cost_usd = Column(Float, nullable=False, default=0.0)
    turn_count = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)

    messages = relationship("AgentMessageModel", back_populates="session",
                            cascade="all, delete-orphan", order_by="AgentMessageModel.created_at")
    tool_executions = relationship("AgentToolExecutionModel", back_populates="session",
                                   cascade="all, delete-orphan")
    children = relationship("AgentSessionModel", backref="parent", remote_side=[id])

    __table_args__ = (
        Index("ix_af_sessions_parent", "parent_session_id"),
        Index("ix_af_sessions_agent_status", "agent_id", "status"),
        Index("ix_af_sessions_tenant_user", "tenant_id", "user_id", "status"),
    )


# ══════════════════════════════════════════════════════════════════════
# Messages
# ══════════════════════════════════════════════════════════════════════

class AgentMessageModel(Base):
    __tablename__ = "af_messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("af_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(JSON, nullable=False, default="")
    tool_calls = Column(JSON, nullable=True)
    tool_call_id = Column(String(64), nullable=True)
    tool_name = Column(String(255), nullable=True)
    token_count = Column(Integer, nullable=False, default=0)
    model = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    is_summarized = Column(Boolean, nullable=False, default=False)

    session = relationship("AgentSessionModel", back_populates="messages")

    __table_args__ = (
        Index("ix_af_messages_session_created", "session_id", "created_at"),
    )


# ══════════════════════════════════════════════════════════════════════
# Tool Executions (Audit Trail)
# ══════════════════════════════════════════════════════════════════════

class AgentToolExecutionModel(Base):
    __tablename__ = "af_tool_executions"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("af_sessions.id"), nullable=False, index=True)
    message_id = Column(String(36), ForeignKey("af_messages.id"), nullable=True)
    tool_call_id = Column(String(64), nullable=False)
    tool_name = Column(String(255), nullable=False, index=True)
    input = Column(JSON, nullable=False, default=dict)
    output = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="success")
    duration_ms = Column(Float, nullable=False, default=0.0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    session = relationship("AgentSessionModel", back_populates="tool_executions")

    __table_args__ = (
        Index("ix_af_tool_exec_session", "session_id", "created_at"),
    )


# ══════════════════════════════════════════════════════════════════════
# Artifacts (agent-generated outputs — reports, data, files)
# ══════════════════════════════════════════════════════════════════════

class AgentArtifactModel(Base):
    __tablename__ = "af_artifacts"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("af_sessions.id"), nullable=False, index=True)
    artifact_type = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    content_json = Column(JSON, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ══════════════════════════════════════════════════════════════════════
# Audit Log
# ══════════════════════════════════════════════════════════════════════

class AgentAuditLogModel(Base):
    __tablename__ = "af_audit_log"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("af_sessions.id"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    actor = Column(String(255), nullable=False, default="")
    action = Column(Text, nullable=False, default="")
    detail = Column(JSON, nullable=True)
    llm_model = Column(String(255), nullable=True)
    llm_input_tokens = Column(Integer, nullable=True)
    llm_output_tokens = Column(Integer, nullable=True)
    llm_latency_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_af_audit_session_time", "session_id", "timestamp"),
    )


# ══════════════════════════════════════════════════════════════════════
# Memory Store (cross-session persistent memory)
# ══════════════════════════════════════════════════════════════════════

class AgentMemoryModel(Base):
    __tablename__ = "af_memory"

    id = Column(String(36), primary_key=True, default=_uuid)
    agent_id = Column(String(255), nullable=False, index=True)
    memory_type = Column(String(50), nullable=False, default="long_term")
    key = Column(String(512), nullable=False)
    value = Column(JSON, nullable=False)
    session_id = Column(String(36), nullable=True)
    ttl_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("ix_af_memory_agent_key", "agent_id", "key", unique=True),
    )
