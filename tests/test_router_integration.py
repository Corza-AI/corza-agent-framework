"""
Tests for the FastAPI router — HTTP-level integration tests.

Uses httpx AsyncClient with ASGITransport to test actual HTTP requests
against the router without starting a server.
"""
import pytest
from fastapi import FastAPI

from corza_agents.api.router import create_agent_router
from corza_agents.api.service import AgentService
from corza_agents.core.types import AgentDefinition
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.tools.registry import ToolRegistry
from tests.helpers import MockLLM, make_text_response

try:
    from httpx import ASGITransport, AsyncClient
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

pytestmark = pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")


async def _build_app(admin_only: bool = True) -> tuple[FastAPI, InMemoryRepository]:
    """Build a FastAPI app with the agent router for testing."""
    llm = MockLLM([make_text_response("Hello from agent")])
    repo = InMemoryRepository()
    await repo.initialize()
    tools = ToolRegistry()

    orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
    await orchestrator.initialize()

    agents = {
        "assistant": AgentDefinition(name="assistant", model="mock:test"),
    }
    service = AgentService(orchestrator, agents)
    router = create_agent_router(orchestrator, agents, service=service, admin_only=admin_only)

    app = FastAPI()
    app.include_router(router, prefix="/api/agent")
    return app, repo


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health returns 200 with status."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/agent/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "ok")


@pytest.mark.asyncio
async def test_create_session_http():
    """POST /sessions creates a session."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent/sessions", json={"agent_id": "assistant"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "idle"


@pytest.mark.asyncio
async def test_get_session_not_found():
    """GET /sessions/{id} returns 404 for missing session."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/agent/sessions/nonexistent-id")
        assert resp.status_code == 404
        data = resp.json()
        assert data.get("error_type") == "not_found"


@pytest.mark.asyncio
async def test_send_message_sync():
    """POST /sessions/{id}/messages with stream=false returns JSON."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create session first
        create_resp = await client.post("/api/agent/sessions", json={"agent_id": "assistant"})
        session_id = create_resp.json()["session_id"]

        # Send message (non-streaming)
        resp = await client.post(
            f"/api/agent/sessions/{session_id}/messages",
            json={"content": "Hi", "stream": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_send_message_sse_format():
    """POST /sessions/{id}/messages with stream=true returns SSE."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/api/agent/sessions", json={"agent_id": "assistant"})
        session_id = create_resp.json()["session_id"]

        resp = await client.post(
            f"/api/agent/sessions/{session_id}/messages",
            json={"content": "Hi", "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Body should contain SSE-formatted events
        body = resp.text
        assert "event:" in body or "data:" in body


@pytest.mark.asyncio
async def test_delete_session_http():
    """DELETE /sessions/{id} removes the session."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/api/agent/sessions", json={"agent_id": "assistant"})
        session_id = create_resp.json()["session_id"]

        resp = await client.delete(f"/api/agent/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"


@pytest.mark.asyncio
async def test_post_tools_blocked_by_default():
    """POST /tools returns 403 when admin_only=True (default)."""
    app, _ = await _build_app(admin_only=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent/tools", json={
            "name": "test_tool",
            "description": "A test",
            "parameters": {},
        })
        assert resp.status_code == 403
        data = resp.json()
        assert "disabled" in data.get("error", "").lower()


@pytest.mark.asyncio
async def test_post_agents_blocked_by_default():
    """POST /agents returns 403 when admin_only=True (default)."""
    app, _ = await _build_app(admin_only=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent/agents", json={
            "name": "new_agent",
            "model": "openai:gpt-4o",
        })
        assert resp.status_code == 403
        data = resp.json()
        assert "disabled" in data.get("error", "").lower()


@pytest.mark.asyncio
async def test_post_tools_allowed_when_opted_in():
    """POST /tools succeeds when admin_only=False."""
    app, _ = await _build_app(admin_only=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent/tools", json={
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "registered"


@pytest.mark.asyncio
async def test_list_agents():
    """GET /agents returns registered agents."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/agent/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert any(a["name"] == "assistant" for a in data)


@pytest.mark.asyncio
async def test_list_tools():
    """GET /tools returns registered tools."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/agent/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_cancel_session_http():
    """POST /sessions/{id}/cancel cancels the session."""
    app, _ = await _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a session first
        create_resp = await client.post("/api/agent/sessions", json={"agent_id": "assistant"})
        session_id = create_resp.json()["session_id"]

        # Cancel it
        resp = await client.post(f"/api/agent/sessions/{session_id}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        assert data["sessions_cancelled"] >= 1
