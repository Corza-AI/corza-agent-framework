"""
Corza Agent Framework — LLM Interface

Provider-agnostic LLM interface with streaming and tool-use support.

Pattern: "provider:model" string selects the backend.
  e.g. "ollama:qwen3:8b", "openai:gpt-5.4", "anthropic:claude-sonnet-4-6"

No default model — user must specify.
"""
import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from corza_agents.core.errors import ContextOverflowError, LLMError, LLMRateLimitError
from corza_agents.core.types import (
    AgentMessage,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
    MessageRole,
    StopReason,
    ToolCall,
    ToolSchema,
)

log = structlog.get_logger("corza_agents.llm")


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider:model_name' into (provider, model_name)."""
    if ":" in model:
        provider, model_name = model.split(":", 1)
        return provider.lower(), model_name
    return "ollama", model


def _messages_to_anthropic(
    messages: list[AgentMessage],
    system_prompt: str,
) -> tuple[list[dict], str | list[dict]]:
    """
    Convert framework messages to Anthropic API format.

    Anthropic requires strict user/assistant alternation.  Consecutive
    TOOL_RESULT messages (from multi-tool turns) are batched into a
    single 'user' message with multiple tool_result content blocks.
    """
    api_messages = []
    pending_tool_results: list[dict] = []

    def _flush_tool_results():
        nonlocal pending_tool_results
        if pending_tool_results:
            api_messages.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            continue
        elif msg.role == MessageRole.USER:
            _flush_tool_results()
            api_messages.append({"role": "user", "content": msg.content})
        elif msg.role == MessageRole.ASSISTANT:
            _flush_tool_results()
            content_blocks: list[dict] = []
            if msg.content:
                text = msg.text() if isinstance(msg.content, list) else msg.content
                if text:
                    content_blocks.append({"type": "text", "text": text})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.tool_name,
                        "input": tc.arguments,
                    })
            api_messages.append({"role": "assistant", "content": content_blocks or msg.content})
        elif msg.role == MessageRole.TOOL_RESULT:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, default=str)
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": content,
            })

    _flush_tool_results()
    system_blocks = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    return api_messages, system_blocks


def _messages_to_openai(
    messages: list[AgentMessage],
    system_prompt: str,
) -> list[dict]:
    """Convert framework messages to OpenAI API format."""
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            continue
        elif msg.role == MessageRole.USER:
            api_messages.append({"role": "user", "content": msg.content if isinstance(msg.content, str) else msg.text()})
        elif msg.role == MessageRole.ASSISTANT:
            m: dict[str, Any] = {"role": "assistant"}
            text = msg.text() if isinstance(msg.content, list) else msg.content
            m["content"] = text or ""
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            api_messages.append(m)
        elif msg.role == MessageRole.TOOL_RESULT:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, default=str)
            api_messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content,
            })
    return api_messages


def _tools_to_anthropic(tools: list[ToolSchema]) -> list[dict]:
    """Convert to Anthropic tool format."""
    result = []
    for t in tools:
        tool_def = {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        result.append(tool_def)
    if result:
        result[-1]["cache_control"] = {"type": "ephemeral"}
    return result


def _tools_to_openai(tools: list[ToolSchema]) -> list[dict]:
    """Convert to OpenAI tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


# ══════════════════════════════════════════════════════════════════════
# OpenAI-compatible provider registry
# Any provider with an OpenAI-compatible API just needs a base URL.
# ══════════════════════════════════════════════════════════════════════

OPENAI_COMPATIBLE_PROVIDERS: dict[str, str] = {
    # Provider name → base URL (empty string = use OpenAI default)
    #
    # Cloud providers
    "openai": "",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "perplexity": "https://api.perplexity.ai",
    "cohere": "https://api.cohere.com/compatibility/v1",
    # Fast inference
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "together": "https://api.together.xyz/v1",
    # Local runners (OpenAI-compatible)
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "jan": "http://localhost:1337/v1",
    "llamacpp": "http://localhost:8080/v1",
    "localai": "http://localhost:8080/v1",
    "lemonade": "http://localhost:8000/api/v1",
    "jellybox": "http://localhost:39281/v1",
    "docker": "http://localhost:12434/engines/v1",
    "vllm": "http://localhost:8000/v1",
    # Cloud platforms (require base_url override via custom_providers)
    "azure": "",
    "bedrock": "",
}


class AgentLLM:
    """
    Provider-agnostic LLM interface with streaming and tool-use support.

    Supports 20+ providers out of the box via OpenAI-compatible API.
    Lazily initializes provider clients. Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        custom_providers: dict[str, str] | None = None,
    ):
        """
        Args:
            api_keys: Optional dict of provider → API key overrides.
                      Falls back to env vars (e.g., OPENAI_API_KEY).
            custom_providers: Optional dict of provider name → base URL
                              for providers not in the built-in registry.
        """
        self._api_keys = api_keys or {}
        self._custom_providers = custom_providers or {}
        self._clients: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def _get_api_key(self, provider: str) -> str:
        """
        Resolve API key for a provider.

        Priority: api_keys dict → {PROVIDER}_API_KEY env var → empty string.
        """
        if provider in self._api_keys:
            return self._api_keys[provider]
        import os
        return os.environ.get(f"{provider.upper()}_API_KEY", "")

    async def _get_client(self, provider: str) -> Any:
        async with self._lock:
            if provider in self._clients:
                return self._clients[provider]
            api_key = self._get_api_key(provider)
            if provider == "anthropic":
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=api_key)
            elif provider == "google":
                from google import genai
                client = genai.Client(api_key=api_key)
            else:
                # OpenAI-compatible providers (OpenAI, Groq, DeepSeek, etc.)
                from openai import AsyncOpenAI
                base_url = (
                    OPENAI_COMPATIBLE_PROVIDERS.get(provider)
                    or self._custom_providers.get(provider)
                    or None  # None = use OpenAI default
                )
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            self._clients[provider] = client
            return client

    # ══════════════════════════════════════════════════════════════════
    # Streaming completion with tool support
    # ══════════════════════════════════════════════════════════════════

    async def stream_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        model: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream a completion with tool-use support.
        Yields LLMStreamChunk objects as they arrive.
        """
        if not model:
            raise LLMError(
                "No model specified. Set 'model' on AgentDefinition "
                "(e.g. 'openai:gpt-5.4', 'anthropic:claude-sonnet-4-6', 'ollama:qwen3:8b').",
                provider="", model="", retryable=False,
            )
        provider, model_name = _parse_model_string(model)

        if provider == "anthropic":
            async for chunk in self._stream_anthropic(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            ):
                yield chunk
        elif provider == "google":
            response = await self._complete_google(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            )
            yield LLMStreamChunk(type="complete", text=response.content,
                                 tool_call=None, usage=response.usage,
                                 stop_reason=response.stop_reason)
        else:
            async for chunk in self._stream_openai(
                messages, tools, model_name, system_prompt, temperature, max_tokens, provider
            ):
                yield chunk

    async def complete_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        model: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        """Non-streaming completion — collects full response."""
        if not model:
            raise LLMError(
                "No model specified. Set 'model' on AgentDefinition "
                "(e.g. 'openai:gpt-5.4', 'anthropic:claude-sonnet-4-6', 'ollama:qwen3:8b').",
                provider="", model="", retryable=False,
            )
        provider, model_name = _parse_model_string(model)

        if provider == "anthropic":
            return await self._complete_anthropic(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            )
        elif provider == "google":
            return await self._complete_google(
                messages, tools, model_name, system_prompt, temperature, max_tokens
            )
        else:
            return await self._complete_openai(
                messages, tools, model_name, system_prompt, temperature, max_tokens, provider
            )

    # ══════════════════════════════════════════════════════════════════
    # Anthropic
    # ══════════════════════════════════════════════════════════════════

    async def _stream_anthropic(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens
    ) -> AsyncIterator[LLMStreamChunk]:
        client = await self._get_client("anthropic")
        api_messages, system_blocks = _messages_to_anthropic(messages, system_prompt)
        api_tools = _tools_to_anthropic(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        try:
            async with client.messages.stream(**kwargs) as stream:
                current_tool_id = None
                current_tool_name = None
                accumulated_json = ""

                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool_id = block.id
                            current_tool_name = block.name
                            accumulated_json = ""
                            yield LLMStreamChunk(
                                type="tool_call_start",
                                tool_call_id=current_tool_id,
                                tool_name=current_tool_name,
                            )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield LLMStreamChunk(type="text_delta", text=delta.text)
                        elif delta.type == "thinking_delta":
                            yield LLMStreamChunk(type="thinking_delta", text=delta.thinking)
                        elif delta.type == "input_json_delta":
                            accumulated_json += delta.partial_json
                            yield LLMStreamChunk(
                                type="tool_call_delta",
                                tool_call_id=current_tool_id,
                                arguments_delta=delta.partial_json,
                            )
                    elif event.type == "content_block_stop":
                        if current_tool_id:
                            try:
                                args = json.loads(accumulated_json) if accumulated_json else {}
                            except json.JSONDecodeError:
                                args = {}
                            yield LLMStreamChunk(
                                type="tool_call_end",
                                tool_call=ToolCall(
                                    id=current_tool_id,
                                    tool_name=current_tool_name or "",
                                    arguments=args,
                                ),
                            )
                            current_tool_id = None
                            current_tool_name = None
                            accumulated_json = ""

                final = await stream.get_final_message()
                usage = LLMUsage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                    cache_creation_tokens=getattr(final.usage, "cache_creation_input_tokens", 0),
                    cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0),
                    total_tokens=final.usage.input_tokens + final.usage.output_tokens,
                )
                stop = StopReason.TOOL_USE if final.stop_reason == "tool_use" else StopReason.END_TURN
                yield LLMStreamChunk(type="usage", usage=usage, stop_reason=stop)

        except Exception as e:
            self._handle_provider_error("anthropic", model_name, e)

    async def _complete_anthropic(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens
    ) -> LLMResponse:
        client = await self._get_client("anthropic")
        api_messages, system_blocks = _messages_to_anthropic(messages, system_prompt)
        api_tools = _tools_to_anthropic(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        start = time.time()
        try:
            response = await client.messages.create(**kwargs)
        except Exception as e:
            self._handle_provider_error("anthropic", model_name, e)

        latency = (time.time() - start) * 1000

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    tool_name=block.name,
                    arguments=block.input,
                ))

        stop = StopReason.TOOL_USE if response.stop_reason == "tool_use" else StopReason.END_TURN

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            model=f"anthropic:{model_name}",
            latency_ms=latency,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0),
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # OpenAI-compatible (OpenAI, Cerebras, Together, etc.)
    # ══════════════════════════════════════════════════════════════════

    async def _stream_openai(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens, provider
    ) -> AsyncIterator[LLMStreamChunk]:
        client = await self._get_client(provider)
        api_messages = _messages_to_openai(messages, system_prompt)
        api_tools = _tools_to_openai(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if api_tools:
            kwargs["tools"] = api_tools
            kwargs["tool_choice"] = "auto"

        try:
            log.debug("llm_stream_request", provider=provider, model=model_name,
                       tool_count=len(api_tools) if api_tools else 0,
                       tool_choice=kwargs.get("tool_choice"))
            stream = await client.chat.completions.create(**kwargs)

            tool_call_buffers: dict[int, dict[str, Any]] = {}
            usage_info: LLMUsage | None = None
            has_content = False
            has_tool_calls = False

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    if chunk.usage:
                        usage_info = LLMUsage(
                            input_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        )
                    continue

                delta = choice.delta
                if delta and delta.content:
                    has_content = True
                    yield LLMStreamChunk(type="text_delta", text=delta.content)

                if delta and delta.tool_calls:
                    has_tool_calls = True
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                                "arguments": "",
                            }
                            yield LLMStreamChunk(
                                type="tool_call_start",
                                tool_call_id=tool_call_buffers[idx]["id"],
                                tool_name=tool_call_buffers[idx]["name"],
                            )
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_buffers[idx]["arguments"] += tc_delta.function.arguments
                        if tc_delta.id and not tool_call_buffers[idx]["id"]:
                            tool_call_buffers[idx]["id"] = tc_delta.id

                if choice.finish_reason:
                    for idx, buf in sorted(tool_call_buffers.items()):
                        try:
                            args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield LLMStreamChunk(
                            type="tool_call_end",
                            tool_call=ToolCall(
                                id=buf["id"],
                                tool_name=buf["name"],
                                arguments=args,
                            ),
                        )

                    stop = StopReason.TOOL_USE if choice.finish_reason == "tool_calls" else StopReason.END_TURN
                    yield LLMStreamChunk(
                        type="usage",
                        usage=usage_info or LLMUsage(),
                        stop_reason=stop,
                    )

            log.info("llm_stream_complete", provider=provider, model=model_name,
                      has_content=has_content, has_tool_calls=has_tool_calls,
                      tool_call_count=len(tool_call_buffers))

        except Exception as e:
            self._handle_provider_error(provider, model_name, e)

    async def _complete_openai(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens, provider
    ) -> LLMResponse:
        client = await self._get_client(provider)
        api_messages = _messages_to_openai(messages, system_prompt)
        api_tools = _tools_to_openai(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if api_tools:
            kwargs["tools"] = api_tools
            kwargs["tool_choice"] = "auto"

        start = time.time()
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            self._handle_provider_error(provider, model_name, e)

        latency = (time.time() - start) * 1000
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    tool_name=tc.function.name,
                    arguments=args,
                ))

        stop = StopReason.TOOL_USE if choice.finish_reason == "tool_calls" else StopReason.END_TURN

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            stop_reason=stop,
            model=f"{provider}:{model_name}",
            latency_ms=latency,
            usage=LLMUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # Google Gemini (non-streaming for now)
    # ══════════════════════════════════════════════════════════════════

    async def _complete_google(
        self, messages, tools, model_name, system_prompt, temperature, max_tokens
    ) -> LLMResponse:
        """Google Gemini completion via google-genai SDK."""
        client = await self._get_client("google")
        from google.genai import types

        contents = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg.text() if isinstance(msg.content, list) else msg.content)],
                ))
            elif msg.role == MessageRole.ASSISTANT:
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg.text() if isinstance(msg.content, list) else msg.content)],
                ))

        gemini_tools = []
        if tools:
            function_declarations = []
            for t in tools:
                function_declarations.append(types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                ))
            gemini_tools = [types.Tool(function_declarations=function_declarations)]

        start = time.time()
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    tools=gemini_tools if gemini_tools else None,
                ),
            )
        except Exception as e:
            self._handle_provider_error("google", model_name, e)

        latency = (time.time() - start) * 1000

        text_parts = []
        tool_calls = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    tool_calls.append(ToolCall(
                        tool_name=part.function_call.name,
                        arguments=dict(part.function_call.args) if part.function_call.args else {},
                    ))

        stop = StopReason.TOOL_USE if tool_calls else StopReason.END_TURN
        usage_meta = response.usage_metadata
        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop,
            model=f"google:{model_name}",
            latency_ms=latency,
            usage=LLMUsage(
                input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
                total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
            ),
        )

    # ══════════════════════════════════════════════════════════════════
    # Error handling
    # ══════════════════════════════════════════════════════════════════

    def _handle_provider_error(self, provider: str, model: str, error: Exception) -> None:
        error_str = str(error).lower()

        if "rate" in error_str and "limit" in error_str:
            raise LLMRateLimitError(
                str(error), provider=provider, model=model
            ) from error

        if "context" in error_str and ("length" in error_str or "window" in error_str or "overflow" in error_str):
            raise ContextOverflowError(
                str(error), provider=provider, model=model
            ) from error

        raise LLMError(
            str(error), provider=provider, model=model,
            retryable="timeout" in error_str or "connection" in error_str,
        ) from error
