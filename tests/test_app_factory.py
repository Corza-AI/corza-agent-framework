"""
Tests for the create_app() factory function.

Verifies that the factory returns a fully wired FastAPI app
with the expected routes.
"""
from fastapi import FastAPI

from corza_agents.app import create_app
from corza_agents.core.types import AgentDefinition


def test_create_app_returns_fastapi():
    """create_app() returns a FastAPI instance with routes."""
    app = create_app(
        agents={"test": AgentDefinition(name="test", model="mock:test")},
    )
    assert isinstance(app, FastAPI)
    assert len(app.routes) > 0


def test_create_app_has_agent_routes():
    """create_app() registers /sessions and /health routes."""
    app = create_app(
        agents={"test": AgentDefinition(name="test", model="mock:test")},
    )
    route_paths = [getattr(r, "path", "") for r in app.routes]
    assert any("/sessions" in p for p in route_paths), (
        f"No /sessions route found in {route_paths}"
    )
    assert any("/health" in p for p in route_paths), (
        f"No /health route found in {route_paths}"
    )
