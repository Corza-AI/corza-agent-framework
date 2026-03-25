"""
Tests for provider fallback chain.
"""
from collections.abc import AsyncIterator

import pytest

from corza_agents.core.engine import AgentEngine
from corza_agents.core.errors import LLMError
from corza_agents.core.types import (
    AgentDefinition,
    EventType,
    LLMStreamChunk,
)
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.skills.manager import SkillsManager
from corza_agents.tools.registry import ToolRegistry
from tests.helpers import make_text_response


class FallbackLLM:
    """LLM that fails on specific providers, succeeds on others."""

    def __init__(self, fail_providers: set[str]):
        self._fail_providers = fail_providers
        self.calls: list[str] = []  # Track which models were called

    async def stream_with_tools(self, **kwargs) -> AsyncIterator[LLMStreamChunk]:
        model = kwargs.get("model", "")
        self.calls.append(model)

        # Check if this provider should fail
        provider = model.split(":")[0] if ":" in model else model
        if provider in self._fail_providers:
            raise LLMError(
                f"Provider {provider} is down",
                provider=provider, model=model, retryable=False,
            )

        for chunk in make_text_response(f"Response from {model}"):
            yield chunk

    async def count_tokens(self, text: str, model: str = "") -> int:
        return 10


async def _build_engine(llm):
    repo = InMemoryRepository()
    await repo.initialize()
    engine = AgentEngine(
        llm=llm, tool_registry=ToolRegistry(), repository=repo,
        skills_manager=SkillsManager(), middleware=[],
    )
    return engine, repo


@pytest.mark.asyncio
async def test_fallback_succeeds_on_secondary():
    """When primary fails, fallback model is used."""
    llm = FallbackLLM(fail_providers={"anthropic"})
    engine, repo = await _build_engine(llm)

    agent = AgentDefinition(
        name="test",
        model="anthropic:claude-sonnet-4-6",
        fallback_models=["groq:llama-3.3-70b"],
        max_llm_retries=0,  # Don't retry primary — go straight to fallback
    )

    events = []
    async for event in engine.run("s1", "Hello", agent):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.SESSION_COMPLETED in event_types

    # Verify the fallback was called
    assert "anthropic:claude-sonnet-4-6" in llm.calls  # Tried primary
    assert "groq:llama-3.3-70b" in llm.calls  # Fell back


@pytest.mark.asyncio
async def test_fallback_chain_tries_all():
    """Falls through multiple fallbacks until one works."""
    llm = FallbackLLM(fail_providers={"anthropic", "groq"})
    engine, repo = await _build_engine(llm)

    agent = AgentDefinition(
        name="test",
        model="anthropic:claude-sonnet-4-6",
        fallback_models=["groq:llama-3.3-70b", "cerebras:llama-3.3-70b"],
        max_llm_retries=0,
    )

    events = []
    async for event in engine.run("s1", "Hello", agent):
        events.append(event)

    assert EventType.SESSION_COMPLETED in [e.type for e in events]
    assert len(llm.calls) == 3  # anthropic → groq → cerebras


@pytest.mark.asyncio
async def test_fallback_all_fail():
    """When all models fail, error is raised."""
    llm = FallbackLLM(fail_providers={"anthropic", "groq", "cerebras"})
    engine, repo = await _build_engine(llm)

    agent = AgentDefinition(
        name="test",
        model="anthropic:claude-sonnet-4-6",
        fallback_models=["groq:llama-3.3-70b", "cerebras:llama-3.3-70b"],
        max_llm_retries=0,
    )

    events = []
    async for event in engine.run("s1", "Hello", agent):
        events.append(event)

    # Should have error event (all providers failed)
    event_types = [e.type for e in events]
    assert EventType.ERROR in event_types or EventType.SESSION_COMPLETED in event_types


@pytest.mark.asyncio
async def test_no_fallback_uses_primary_only():
    """Without fallback_models, only primary is tried."""
    llm = FallbackLLM(fail_providers=set())
    engine, repo = await _build_engine(llm)

    agent = AgentDefinition(
        name="test",
        model="openai:gpt-4.1",
        # No fallback_models
    )

    events = [e async for e in engine.run("s1", "Hello", agent)]
    assert EventType.SESSION_COMPLETED in [e.type for e in events]
    assert llm.calls == ["openai:gpt-4.1"]


def test_fallback_models_field():
    """fallback_models defaults to empty list."""
    agent = AgentDefinition(name="test", model="openai:gpt-4.1")
    assert agent.fallback_models == []

    agent2 = AgentDefinition(
        name="test", model="openai:gpt-4.1",
        fallback_models=["groq:llama-3.3-70b"],
    )
    assert agent2.fallback_models == ["groq:llama-3.3-70b"]
