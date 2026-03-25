"""Tests for the LLM interface."""
import pytest

from corza_agents import AgentLLM
from corza_agents.core.errors import LLMError


@pytest.mark.asyncio
async def test_empty_model_raises():
    llm = AgentLLM()
    with pytest.raises(LLMError, match="No model specified"):
        async for _ in llm.stream_with_tools([], [], "", "", 0.0, 100):
            pass


@pytest.mark.asyncio
async def test_empty_model_complete_raises():
    llm = AgentLLM()
    with pytest.raises(LLMError, match="No model specified"):
        await llm.complete_with_tools([], [], "", "", 0.0, 100)


def test_parse_model_string():
    from corza_agents.core.llm import _parse_model_string

    assert _parse_model_string("openai:gpt-4o") == ("openai", "gpt-4o")
    assert _parse_model_string("anthropic:claude-3-haiku") == ("anthropic", "claude-3-haiku")
    assert _parse_model_string("ollama:qwen3.5:9b") == ("ollama", "qwen3.5:9b")
    # No prefix — splits on first colon (ambiguous, use explicit prefix)
    assert _parse_model_string("qwen3.5:9b") == ("qwen3.5", "9b")
    # Simple model name without colon defaults to ollama
    assert _parse_model_string("llama3") == ("ollama", "llama3")


def test_api_key_from_dict():
    llm = AgentLLM(api_keys={"openai": "sk-test"})
    assert llm._get_api_key("openai") == "sk-test"
