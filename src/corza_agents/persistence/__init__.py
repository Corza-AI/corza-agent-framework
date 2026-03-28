"""
Corza Agent Framework — Persistence Layer

Backend support: memory (testing), sqlite (dev), postgres (production).
"""

from corza_agents.persistence.base import BaseRepository
from corza_agents.persistence.factory import create_repository
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.persistence.models import (
    AgentArtifactModel,
    AgentAuditLogModel,
    AgentMemoryModel,
    AgentMessageModel,
    AgentSessionModel,
    AgentToolExecutionModel,
    Base,
)

# PostgresRepository requires asyncpg + sqlalchemy
try:
    from corza_agents.persistence.repository import PostgresRepository, Repository
except ImportError:

    def _postgres_not_installed(*args, **kwargs):
        raise ImportError(
            "PostgresRepository requires asyncpg and sqlalchemy. "
            "Install with: pip install 'corza-agents[postgres]'"
        )

    PostgresRepository = _postgres_not_installed  # type: ignore[assignment,misc]
    Repository = _postgres_not_installed  # type: ignore[assignment,misc]

# SQLiteRepository requires aiosqlite
try:
    from corza_agents.persistence.sqlite import SQLiteRepository
except ImportError:

    def _sqlite_not_installed(*args, **kwargs):
        raise ImportError(
            "SQLiteRepository requires aiosqlite. Install with: pip install 'corza-agents[sqlite]'"
        )

    SQLiteRepository = _sqlite_not_installed  # type: ignore[assignment,misc]

__all__ = [
    "BaseRepository",
    "create_repository",
    "InMemoryRepository",
    "PostgresRepository",
    "Repository",
    "SQLiteRepository",
    "Base",
    "AgentSessionModel",
    "AgentMessageModel",
    "AgentToolExecutionModel",
    "AgentArtifactModel",
    "AgentAuditLogModel",
    "AgentMemoryModel",
]
