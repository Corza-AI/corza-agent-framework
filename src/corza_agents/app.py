"""
Corza Agent Framework — FastAPI App Factory

Create a fully wired FastAPI app in one function call.

Usage:
    from corza_agents import create_app, AgentDefinition

    assistant = AgentDefinition(name="assistant", model="openai:gpt-5.4", tools=["greet"])
    app = create_app(agents={"assistant": assistant})
    # Run with: uvicorn myapp:app
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from corza_agents.api.router import create_agent_router
from corza_agents.api.service import AgentService
from corza_agents.core.llm import AgentLLM
from corza_agents.core.types import AgentDefinition
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.persistence.factory import create_repository
from corza_agents.tools.registry import ToolRegistry

log = structlog.get_logger("corza_agents.app")


def create_app(
    agents: dict[str, AgentDefinition],
    db_url: str | None = None,
    api_keys: dict[str, str] | None = None,
    tool_registry: ToolRegistry | None = None,
    cors_origins: list[str] | None = None,
    prefix: str = "/api/agent",
    title: str = "Corza Agent API",
    admin_only: bool = True,
    **fastapi_kwargs: Any,
) -> FastAPI:
    """
    Create a FastAPI app with the agent framework fully wired.

    Args:
        agents: Named agent definitions (e.g., {"assistant": assistant_def}).
        db_url: Database URL. PostgreSQL for production, SQLite if None.
        api_keys: LLM provider API keys (e.g., {"openai": "sk-..."}).
        tool_registry: Pre-configured ToolRegistry with your tools registered.
        cors_origins: Allowed CORS origins (e.g., ["http://localhost:3000"]).
        prefix: URL prefix for agent endpoints (default "/api/agent").
        title: FastAPI app title.
        admin_only: If True (default), POST /tools and POST /agents return 403.

    Returns:
        A configured FastAPI application. Run with: uvicorn module:app
    """
    # Build components eagerly (validation happens here)
    repo = _resolve_repo(db_url)
    llm = AgentLLM(api_keys=api_keys)
    tools = tool_registry or ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    service = AgentService(orchestrator, agents)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        log.info("corza_startup", agents=list(agents.keys()))
        await orchestrator.initialize()
        app.state.agent_service = service
        app.state.orchestrator = orchestrator
        yield
        await orchestrator.shutdown()
        log.info("corza_shutdown")

    app = FastAPI(title=title, lifespan=lifespan, **fastapi_kwargs)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the agent router
    router = create_agent_router(orchestrator, agents, service=service, admin_only=admin_only)
    app.include_router(router, prefix=prefix)

    return app


def _resolve_repo(db_url: str | None):
    """Pick the right persistence backend from a URL string."""
    if not db_url:
        return create_repository("sqlite", db_path="corza_agents.db")
    if "postgres" in db_url or "postgresql" in db_url:
        return create_repository("postgres", db_url=db_url)
    if "sqlite" in db_url:
        path = db_url.split("///", 1)[-1] if "///" in db_url else db_url
        return create_repository("sqlite", db_path=path)
    return create_repository("memory")
