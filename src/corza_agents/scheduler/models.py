"""
Corza Agent Framework — Scheduler DB Models

Stores scheduled agent runs in PostgreSQL.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Index, Integer, String, Text

from corza_agents.persistence.models import Base


def _utcnow():
    return datetime.now(UTC)


def _uuid():
    return str(uuid.uuid4())


class AgentScheduleModel(Base):
    __tablename__ = "af_schedules"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    agent_id = Column(String(255), nullable=False, index=True)
    tenant_id = Column(String(36), nullable=True, index=True)
    user_id = Column(String(36), nullable=True)
    schedule_type = Column(String(20), nullable=False)  # cron, once, event
    cron_expression = Column(String(100), nullable=True)
    run_at = Column(DateTime(timezone=True), nullable=True)
    event_type = Column(String(255), nullable=True)
    prompt_template = Column(Text, nullable=False, default="")
    agent_config = Column(JSON, nullable=False, default=dict)
    variables = Column(JSON, nullable=False, default=dict)
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)
    enabled = Column(Boolean, nullable=False, default=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    last_run_session_id = Column(String(36), nullable=True)
    last_run_status = Column(String(20), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True)
    run_count = Column(Integer, nullable=False, default=0)
    max_runs = Column(Integer, nullable=True)  # None = unlimited
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("ix_af_schedules_next_run", "enabled", "next_run_at"),
        Index("ix_af_schedules_event", "enabled", "event_type"),
    )
