"""
Corza Agent Framework — Repository Factory

Create a persistence backend by name.

Usage:
    from corza_agents import create_repository

    # Testing (no deps needed)
    repo = create_repository("memory")

    # Local dev
    repo = create_repository("sqlite", db_path="agents.db")

    # Production
    repo = create_repository("postgres", db_url="postgresql+asyncpg://...")
"""
from corza_agents.persistence.base import BaseRepository


def create_repository(backend: str = "memory", **kwargs) -> BaseRepository:
    """
    Factory for persistence backends.

    Args:
        backend: One of "memory", "sqlite", "postgres".
        **kwargs: Backend-specific arguments:
            - memory: (no args)
            - sqlite: db_path (str, default "corza_agents.db")
            - postgres: db_url (str), echo (bool)

    Returns:
        A BaseRepository instance (call .initialize() before use).
    """
    backend = backend.lower().strip()

    if backend == "memory":
        from corza_agents.persistence.memory import InMemoryRepository
        return InMemoryRepository()

    elif backend == "sqlite":
        from corza_agents.persistence.sqlite import SQLiteRepository
        return SQLiteRepository(**kwargs)

    elif backend in ("postgres", "postgresql"):
        from corza_agents.persistence.repository import PostgresRepository
        return PostgresRepository(**kwargs)

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Choose from: memory, sqlite, postgres"
        )
