"""
Security tests for the Corza Agent Framework.

Covers:
- Admin-only API endpoint gating (POST /tools, POST /agents)
- Code execution env-var safety gate (CORZA_ALLOW_CODE_EXECUTION)
"""
import os

import httpx
import pytest
from fastapi import FastAPI

from corza_agents.api.router import create_agent_router
from corza_agents.core.types import (
    ExecutionContext,
    RegisteredTool,
    ToolType,
)
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.tools.handlers import execute_code_tool
from corza_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


async def _make_app(admin_only: bool = True) -> FastAPI:
    """Build a minimal FastAPI app with the agent router mounted."""
    llm = MockLLM([make_text_response("Ok.")])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    app = FastAPI()
    router = create_agent_router(orchestrator, admin_only=admin_only)
    app.include_router(router)
    return app


# ══════════════════════════════════════════════════════════════════════
# Admin-only endpoint tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_post_tools_blocked_by_default():
    """POST /tools returns 403 when admin_only=True (default)."""
    app = await _make_app(admin_only=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post("/tools", json={
            "name": "sneaky_tool",
            "description": "Should not be registered",
        })

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_post_agents_blocked_by_default():
    """POST /agents returns 403 when admin_only=True (default)."""
    app = await _make_app(admin_only=True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post("/agents", json={
            "name": "sneaky_agent",
            "description": "Should not be registered",
            "model": "mock:test",
        })

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_post_tools_allowed_when_opted_in():
    """POST /tools returns 200 when admin_only=False."""
    app = await _make_app(admin_only=False)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post("/tools", json={
            "name": "allowed_tool",
            "description": "This tool should be registered",
        })

    assert resp.status_code == 200
    assert resp.json()["status"] == "registered"


# ══════════════════════════════════════════════════════════════════════
# Code execution safety gate tests
# ══════════════════════════════════════════════════════════════════════


def _code_tool_def() -> RegisteredTool:
    """Minimal RegisteredTool for code execution tests."""
    return RegisteredTool(
        name="test_code",
        description="Test code executor",
        tool_type=ToolType.CODE,
        config={"timeout": 10},
    )


def _code_context() -> ExecutionContext:
    return ExecutionContext(
        session_id="code-session-1",
        agent_id="agent-code",
        agent_name="coder",
    )


@pytest.mark.asyncio
async def test_code_execution_blocked_without_env():
    """Code execution is blocked when CORZA_ALLOW_CODE_EXECUTION is not set."""
    # Ensure the env var is NOT set
    env_backup = os.environ.pop("CORZA_ALLOW_CODE_EXECUTION", None)
    try:
        result = await execute_code_tool(
            tool_def=_code_tool_def(),
            args={"code": "print('hi')"},
            context=_code_context(),
        )

        assert result["status"] == "error"
        assert "CORZA_ALLOW_CODE_EXECUTION" in result["message"]
    finally:
        if env_backup is not None:
            os.environ["CORZA_ALLOW_CODE_EXECUTION"] = env_backup


@pytest.mark.asyncio
async def test_code_execution_allowed_with_env():
    """Code execution succeeds when CORZA_ALLOW_CODE_EXECUTION=true."""
    env_backup = os.environ.get("CORZA_ALLOW_CODE_EXECUTION")
    os.environ["CORZA_ALLOW_CODE_EXECUTION"] = "true"
    try:
        result = await execute_code_tool(
            tool_def=_code_tool_def(),
            args={"code": "import json; print(json.dumps({'answer': 42}))"},
            context=_code_context(),
        )

        assert result["status"] == "success"
    finally:
        if env_backup is not None:
            os.environ["CORZA_ALLOW_CODE_EXECUTION"] = env_backup
        else:
            os.environ.pop("CORZA_ALLOW_CODE_EXECUTION", None)
