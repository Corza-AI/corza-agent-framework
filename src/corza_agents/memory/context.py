"""
Corza Agent Framework — Context Window Manager

Manages the context window to prevent overflow.
3-layer defense inspired by Deep Agents:
  1. Truncate old tool arguments in messages
  2. Auto-summarize old messages via LLM
  3. Manual compact tool available to agent

All operations are DB-backed: old messages are marked as summarized
(not deleted), and the summary is injected as a new message.
"""
import json
from typing import Any

import structlog

from corza_agents.core.types import (
    AgentMessage,
    MessageRole,
    _now,
    _uuid,
)
from corza_agents.persistence.base import BaseRepository

log = structlog.get_logger("corza_agents.context")

# Rough token estimation: ~4 chars per token for English
CHARS_PER_TOKEN = 4

# Default context window limits by model name (after provider: prefix).
# Override at runtime via ContextManager(custom_context_limits={...}).
_DEFAULT_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-4.1": 1_048_576,
    "gpt-4.1-mini": 1_048_576,
    "gpt-4.1-nano": 1_048_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Anthropic
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    # Google
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    # Open-source / local
    "qwen3:8b": 32_000,
    "llama-3.3-70b": 128_000,
    "deepseek-chat": 128_000,
}

DEFAULT_CONTEXT_LIMIT = 128_000


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(msg: AgentMessage) -> int:
    if msg.token_count > 0:
        return msg.token_count
    content = msg.content
    if isinstance(content, list):
        content = json.dumps(content, default=str)
    elif not isinstance(content, str):
        content = str(content)
    tokens = estimate_tokens(content)
    if msg.tool_calls:
        tokens += estimate_tokens(json.dumps([tc.model_dump() for tc in msg.tool_calls]))
    return tokens


def get_context_limit(model: str, custom_limits: dict[str, int] | None = None) -> int:
    """
    Get context window size for a model.

    Checks custom_limits first, then built-in defaults, then DEFAULT_CONTEXT_LIMIT.
    """
    if ":" in model:
        _, model_name = model.split(":", 1)
    else:
        model_name = model

    if custom_limits and model_name in custom_limits:
        return custom_limits[model_name]
    return _DEFAULT_MODEL_CONTEXT_LIMITS.get(model_name, DEFAULT_CONTEXT_LIMIT)


class ContextManager:
    """
    Manages the conversation context to fit within model limits.

    Strategy:
    1. Load messages from DB for the session
    2. Estimate total tokens
    3. If > trigger_threshold (default 80% of limit):
       a. Layer 1: Truncate old tool arguments
       b. Layer 2: Auto-summarize old messages, keeping recent ones
    4. Return optimized message list for the LLM call
    """

    def __init__(
        self,
        repository: BaseRepository,
        trigger_fraction: float = 0.80,
        keep_recent_fraction: float = 0.15,
        custom_context_limits: dict[str, int] | None = None,
    ):
        self._repo = repository
        self._trigger_fraction = trigger_fraction
        self._keep_recent_fraction = keep_recent_fraction
        self._custom_limits = custom_context_limits

    async def build_context(
        self,
        session_id: str,
        system_prompt: str,
        model: str,
        llm: Any | None = None,
        force_compact: bool = False,
    ) -> list[AgentMessage]:
        """
        Build an optimized message list that fits in the model's context window.

        Args:
            session_id: The session to load messages for
            system_prompt: The full system prompt (already built)
            model: Model string for context limit lookup
            llm: AgentLLM instance for summarization (if needed)
            force_compact: If True, skip threshold check and compact immediately
                (triggered by the agent calling manage_context)

        Returns:
            List of AgentMessage objects ready for the LLM call
        """
        messages = await self._repo.get_messages(session_id, include_summarized=False)
        if not messages:
            return messages

        context_limit = get_context_limit(model, self._custom_limits)
        system_tokens = estimate_tokens(system_prompt)
        trigger_tokens = int(context_limit * self._trigger_fraction)

        total_tokens = system_tokens + sum(estimate_message_tokens(m) for m in messages)

        if total_tokens <= trigger_tokens and not force_compact:
            return messages

        log.info("context_over_threshold",
                 session_id=session_id,
                 total_tokens=total_tokens,
                 trigger=trigger_tokens,
                 message_count=len(messages))

        # Layer 1: Truncate old tool call arguments
        messages = self._truncate_old_tool_args(messages, keep_recent=5)
        total_tokens = system_tokens + sum(estimate_message_tokens(m) for m in messages)

        if total_tokens <= trigger_tokens:
            log.info("context_fixed_by_truncation", total_tokens=total_tokens)
            return messages

        # Layer 2: Auto-summarize old messages
        if llm:
            messages = await self._auto_summarize(
                session_id=session_id,
                messages=messages,
                system_tokens=system_tokens,
                target_tokens=int(context_limit * 0.5),
                llm=llm,
                model=model,
            )

        return messages

    def _truncate_old_tool_args(
        self,
        messages: list[AgentMessage],
        keep_recent: int = 5,
        max_arg_chars: int = 500,
    ) -> list[AgentMessage]:
        """
        Layer 1: Truncate large tool call arguments in old messages.
        Keeps recent messages untouched.
        """
        if len(messages) <= keep_recent:
            return messages

        truncated = []
        cutoff = len(messages) - keep_recent

        for i, msg in enumerate(messages):
            if i < cutoff and msg.tool_calls:
                new_calls = []
                for tc in msg.tool_calls:
                    args = tc.arguments
                    truncated_args = {}
                    for k, v in args.items():
                        val_str = json.dumps(v, default=str) if not isinstance(v, str) else v
                        if len(val_str) > max_arg_chars:
                            truncated_args[k] = val_str[:max_arg_chars] + "...(truncated)"
                        else:
                            truncated_args[k] = v
                    new_calls.append(tc.model_copy(update={"arguments": truncated_args}))
                msg = msg.model_copy(update={"tool_calls": new_calls, "token_count": 0})

            if i < cutoff and msg.role == MessageRole.TOOL_RESULT:
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, default=str)
                if len(content) > max_arg_chars * 4:
                    short = content[:max_arg_chars * 2] + "\n...(truncated)..." + content[-500:]
                    msg = msg.model_copy(update={"content": short, "token_count": 0})

            truncated.append(msg)

        return truncated

    async def _auto_summarize(
        self,
        session_id: str,
        messages: list[AgentMessage],
        system_tokens: int,
        target_tokens: int,
        llm: Any,
        model: str,
    ) -> list[AgentMessage]:
        """
        Layer 2: Summarize old messages via LLM, keep recent ones intact.

        Process:
        1. Determine cutoff: keep the most recent N messages
        2. Summarize everything before the cutoff
        3. Mark summarized messages in DB
        4. Return [summary_message, ...recent_messages]
        """
        keep_count = max(3, int(len(messages) * self._keep_recent_fraction))
        if keep_count >= len(messages):
            return messages

        to_summarize = messages[:-keep_count]
        to_keep = messages[-keep_count:]

        summary_text = await self._generate_summary(to_summarize, llm, model)

        summarized_ids = [m.id for m in to_summarize]
        await self._repo.mark_messages_summarized(session_id, summarized_ids)

        summary_msg = AgentMessage(
            id=_uuid(),
            session_id=session_id,
            role=MessageRole.USER,
            content=f"[CONVERSATION SUMMARY — {len(to_summarize)} messages summarized]\n\n{summary_text}",
            created_at=_now(),
        )
        await self._repo.add_message(summary_msg)

        log.info("context_summarized",
                 session_id=session_id,
                 messages_summarized=len(to_summarize),
                 messages_kept=len(to_keep),
                 summary_tokens=estimate_tokens(summary_text))

        return [summary_msg] + to_keep

    async def _generate_summary(
        self,
        messages: list[AgentMessage],
        llm: Any,
        model: str,
    ) -> str:
        """Generate a summary of the given messages using the LLM."""
        conversation_text = []
        for msg in messages:
            role = msg.role.value.upper()
            text = msg.text() if isinstance(msg.content, list) else str(msg.content)
            if msg.tool_calls:
                tools_desc = ", ".join(tc.tool_name for tc in msg.tool_calls)
                text += f" [Called tools: {tools_desc}]"
            if len(text) > 2000:
                text = text[:1000] + "..." + text[-500:]
            conversation_text.append(f"[{role}] {text}")

        prompt = (
            "Summarize the following conversation concisely. "
            "Preserve key facts, decisions, tool results, and findings. "
            "Omit verbose tool outputs and intermediate reasoning.\n\n"
            + "\n\n".join(conversation_text)
        )

        from corza_agents.core.types import LLMResponse
        summary_messages = [
            AgentMessage(role=MessageRole.USER, content=prompt)
        ]
        response: LLMResponse = await llm.complete_with_tools(
            messages=summary_messages,
            tools=[],
            model=model,
            system_prompt="You are a conversation summarizer. Be concise and factual.",
            temperature=0.0,
            max_tokens=2000,
        )
        return response.content
