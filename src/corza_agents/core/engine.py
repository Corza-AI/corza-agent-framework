"""
Corza Agent Framework — ReAct Agent Engine

The core agent loop. Provider-agnostic. DB-backed. Streaming.

Flow:
  1. Load/create session from DB
  2. Build context: system_prompt + skills + memory + messages
  3. Call LLM with tools (streaming) — with retry on transient errors
  4. If tool_calls → execute tools → persist → loop back to 3
  5. If end_turn → persist → finalize
  6. Stream events throughout

Production-grade ReAct loop implementation.
"""
import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import structlog

from corza_agents.core.errors import (
    ContextOverflowError,
    LLMError,
    LLMRateLimitError,
)
from corza_agents.core.llm import AgentLLM
from corza_agents.core.types import (
    AgentDefinition,
    AgentMessage,
    AgentSession,
    ExecutionContext,
    LLMResponse,
    LLMUsage,
    MessageRole,
    SessionStatus,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolStatus,
)
from corza_agents.memory.context import ContextManager
from corza_agents.memory.health import (
    CONTEXT_HARD_STOP_MESSAGE,
    CONTEXT_WARNING_MESSAGE,
    ContextHealthConfig,
    assess_health,
)
from corza_agents.memory.working import WorkingMemory
from corza_agents.middleware.base import BaseMiddleware
from corza_agents.persistence.base import BaseRepository
from corza_agents.prompts.templates import build_system_prompt
from corza_agents.skills.manager import SkillsManager
from corza_agents.streaming.events import (
    StreamEvent,
    error_event,
    session_completed,
    session_started,
    text_delta,
    thinking_delta,
    tool_call_event,
    tool_executing,
    tool_result_event,
    turn_completed,
    turn_started,
)
from corza_agents.tools.registry import ToolRegistry

log = structlog.get_logger("corza_agents.engine")


def _json_safe(obj: Any) -> Any:
    """Recursively strip non-JSON-serializable values from a dict/list tree."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            safe = _json_safe(v)
            if safe is not _SKIP:
                out[k] = safe
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj if _json_safe(x) is not _SKIP]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        json.dumps(obj, default=str)
        return str(obj)
    except (TypeError, ValueError):
        return _SKIP

_SKIP = object()


class AgentEngine:
    """
    The ReAct agent loop engine.

    Orchestrates: LLM calls → tool execution → message persistence → streaming.
    Stateless between runs — all state lives in the repository.

    Session locking: Only one run() can execute per session_id at a time.
    Concurrent requests to the same session wait in line.
    """

    def __init__(
        self,
        llm: AgentLLM,
        tool_registry: ToolRegistry,
        repository: BaseRepository,
        skills_manager: SkillsManager | None = None,
        middleware: list[BaseMiddleware] | None = None,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._repo = repository
        self._skills = skills_manager or SkillsManager()
        self._middleware = middleware or []
        self._context_manager = ContextManager(repository)
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._cancelled_sessions: set[str] = set()

    @property
    def repository(self) -> BaseRepository:
        """Public accessor for the persistence repository."""
        return self._repo

    async def cancel(self, session_id: str, cascade: bool = True) -> int:
        """
        Nuclear stop — cancel a running session and optionally all its children.

        Args:
            session_id: The session to cancel.
            cascade: If True (default), also cancel all child sessions
                     (sub-agents spawned by this session).

        Returns:
            Number of sessions cancelled (including children).
        """
        cancelled_count = 0

        # Mark for cancellation — the run loop checks this at the top of each turn
        self._cancelled_sessions.add(session_id)
        await self._repo.update_session(
            session_id, status=SessionStatus.CANCELLED, error="Cancelled by user",
        )
        cancelled_count += 1
        log.info("session_cancelled", session_id=session_id, cascade=cascade)

        if cascade:
            # Find and cancel all child sessions
            children = await self._repo.get_child_sessions(session_id)
            for child in children:
                child_id = child.id if hasattr(child, "id") else child.get("id", "")
                if child_id:
                    self._cancelled_sessions.add(child_id)
                    await self._repo.update_session(
                        child_id, status=SessionStatus.CANCELLED,
                        error="Cancelled (parent cancelled)",
                    )
                    cancelled_count += 1
                    log.info("child_session_cancelled",
                             child_id=child_id, parent_id=session_id)

        return cancelled_count

    async def run(
        self,
        session_id: str,
        user_message: str,
        agent_def: AgentDefinition,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute the agent loop for a session.

        Creates or resumes a session, adds the user message, and runs
        the ReAct loop until completion or max_turns.

        Yields StreamEvent objects for real-time progress tracking.
        """
        # Acquire per-session lock to prevent concurrent writes
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        session_lock = self._session_locks[session_id]

        async with session_lock:
            async for event in self._run_locked(
                session_id, user_message, agent_def, metadata, variables,
            ):
                yield event

        # Clean up lock if no longer needed
        if session_id in self._session_locks and not session_lock.locked():
            self._session_locks.pop(session_id, None)

    async def _run_locked(
        self,
        session_id: str,
        user_message: str,
        agent_def: AgentDefinition,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Internal: execute the agent loop under session lock."""
        session = await self._ensure_session(session_id, agent_def, metadata)
        working_memory = WorkingMemory(session_id=session_id, metadata=metadata or {})

        context = ExecutionContext(
            session_id=session_id,
            agent_id=agent_def.id,
            agent_name=agent_def.name,
            parent_session_id=session.parent_session_id,
            user_id=session.user_id or (metadata or {}).get("user_id", ""),
            tenant_id=session.tenant_id or (metadata or {}).get("tenant_id", ""),
            metadata=metadata or {},
            working_memory=working_memory,
            repository=self._repo,
        )

        # Persist user message
        user_msg = AgentMessage(
            session_id=session_id,
            role=MessageRole.USER,
            content=user_message,
        )
        await self._repo.add_message(user_msg)
        await self._repo.update_session(session_id, status=SessionStatus.RUNNING)

        yield session_started(session_id, agent_def.id, agent_def.name)

        # Resolve skills (including runtime-injected) and build tool list
        skills = self._skills.resolve(agent_def.skills)
        if context.active_skills:
            skills = skills + list(context.active_skills)
        skill_tools = self._skills.get_required_tools(skills)
        all_tool_names = list(set(agent_def.tools + skill_tools))
        tool_schemas = self._tools.get_schemas(all_tool_names)

        turn = 0
        total_usage = LLMUsage()
        llm_response = LLMResponse()

        try:
            while turn < agent_def.max_turns:
                # Nuclear stop check — exit immediately if cancelled
                if session_id in self._cancelled_sessions:
                    self._cancelled_sessions.discard(session_id)
                    log.info("session_cancel_detected", session_id=session_id, turn=turn)
                    await self._repo.update_session(
                        session_id, status=SessionStatus.CANCELLED,
                        error="Cancelled by user",
                    )
                    yield error_event(session_id, "cancelled",
                                      "Session cancelled by user.", turn)
                    return

                turn += 1
                context.turn_number = turn
                yield turn_started(session_id, turn)

                # Check if agent requested manual compaction on a prior turn
                force_compact = bool(working_memory.get("_compact_requested"))
                if force_compact:
                    working_memory.store("_compact_requested", False)
                    log.info("manual_compaction_triggered", session_id=session_id, turn=turn)

                # Load persistent data from DB for prompt visibility
                knowledge_index = await self._load_knowledge_index(agent_def.id)
                skill_index = await self._load_skill_index(agent_def.id)
                objective = await self._load_objective(agent_def.id)
                plan = working_memory.get("_plan") or []

                # Build context (system prompt + messages, with compaction)
                system_prompt = build_system_prompt(
                    agent_def, skills=skills,
                    working_memory_context=working_memory.get_context_for_llm(),
                    variables=variables or {},
                    registered_tools=self._tools.get_schemas(all_tool_names) if all_tool_names else None,
                    knowledge_index=knowledge_index,
                    skill_index=skill_index,
                    objective=objective,
                    plan=plan,
                    turn_number=turn,
                    max_turns=agent_def.max_turns,
                )
                messages = await self._context_manager.build_context(
                    session_id, system_prompt, agent_def.model, llm=self._llm,
                    force_compact=force_compact,
                )

                # Run middleware: before_llm_call
                current_messages, current_tools = messages, tool_schemas
                for mw in self._middleware:
                    current_messages, current_tools = await mw.before_llm_call(
                        current_messages, current_tools, context
                    )

                # Call LLM with retry — text/thinking deltas are streamed
                # in real-time via an async queue, yielded as they arrive.
                _stream_q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
                _llm_error: list[Exception] = []  # Capture errors from the LLM task

                async def _on_stream(evt: StreamEvent):
                    await _stream_q.put(evt)

                async def _llm_task():
                    try:
                        return await self._call_llm_with_retry(
                            session_id=session_id,
                            messages=current_messages,
                            tools=current_tools,
                            agent_def=agent_def,
                            system_prompt=system_prompt,
                            turn=turn,
                            on_stream_event=_on_stream,
                        )
                    except Exception as e:
                        _llm_error.append(e)
                        raise
                    finally:
                        await _stream_q.put(None)  # Sentinel

                llm_future = asyncio.ensure_future(_llm_task())

                # Yield streaming events as they arrive
                while True:
                    evt = await _stream_q.get()
                    if evt is None:
                        break
                    yield evt

                # Re-raise any LLM error
                try:
                    full_text, tool_calls, usage, stop_reason = await llm_future
                except (LLMError, ContextOverflowError) as e:
                    # Turn-level recovery: don't fail the session, allow resume
                    log.error("turn_llm_failed", session_id=session_id, turn=turn,
                              error=str(e)[:500])
                    yield error_event(session_id, "turn_error", str(e)[:1000], turn,
                                      recoverable=True)
                    await self._repo.update_session(
                        session_id, status=SessionStatus.WAITING_INPUT,
                        error=str(e)[:2000],
                    )
                    for mw in self._middleware:
                        await mw.on_error(e, context)
                    break

                # Accumulate usage
                total_usage.input_tokens += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                total_usage.total_tokens += usage.total_tokens
                total_usage.cache_creation_tokens += usage.cache_creation_tokens
                total_usage.cache_read_tokens += usage.cache_read_tokens

                # Run middleware: after_llm_call
                llm_response = LLMResponse(
                    content=full_text, tool_calls=tool_calls,
                    stop_reason=stop_reason, usage=usage, model=agent_def.model,
                )
                for mw in self._middleware:
                    llm_response = await mw.after_llm_call(llm_response, context)

                # Persist assistant message
                assistant_msg = AgentMessage(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=llm_response.content,
                    tool_calls=llm_response.tool_calls if llm_response.tool_calls else None,
                    token_count=usage.output_tokens,
                    model=agent_def.model,
                )
                await self._repo.add_message(assistant_msg)

                # No tool calls → agent is done (natural stop)
                if not llm_response.tool_calls:
                    yield turn_completed(session_id, turn, stop_reason.value)
                    for mw in self._middleware:
                        await mw.on_turn_complete(turn, context)
                    break

                # Execute tool calls
                for tc in llm_response.tool_calls:
                    yield tool_executing(session_id, tc.tool_name, tc.id, turn)

                    # Middleware: before_tool_call
                    current_tc: ToolCall | None = tc
                    for mw in self._middleware:
                        if current_tc is None:
                            break
                        current_tc = await mw.before_tool_call(current_tc, context)

                    if current_tc is None:
                        result = ToolResult(
                            tool_call_id=tc.id, tool_name=tc.tool_name,
                            status=ToolStatus.DENIED, error="Blocked by middleware",
                        )
                    else:
                        result = await self._tools.execute(current_tc, context)

                    # Middleware: after_tool_call
                    for mw in self._middleware:
                        result = await mw.after_tool_call(tc, result, context)

                    # Store tool result in working memory
                    if result.status == ToolStatus.SUCCESS and result.output:
                        working_memory.store(f"tool:{tc.tool_name}:{tc.id[:8]}", result.output)

                    # Persist tool execution
                    await self._repo.log_tool_execution(
                        session_id=session_id,
                        message_id=assistant_msg.id,
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                        input_data=tc.arguments,
                        output_data=result.output,
                        status=result.status.value,
                        duration_ms=result.duration_ms,
                        error=result.error,
                    )

                    # Persist tool result as a message
                    output_content = result.output
                    if isinstance(output_content, (dict, list)):
                        output_content = json.dumps(output_content, default=str)
                    elif output_content is None:
                        output_content = result.error or "No output"
                    tool_msg = AgentMessage(
                        session_id=session_id,
                        role=MessageRole.TOOL_RESULT,
                        content=str(output_content),
                        tool_call_id=tc.id,
                        tool_name=tc.tool_name,
                    )
                    await self._repo.add_message(tool_msg)

                    output_preview = str(output_content) if output_content else ""
                    # Extract card_data for frontend tool card rendering.
                    # Tools return ToolOutput with card_data dict containing
                    # structured results (query, code, results, etc.)
                    _card_data = None
                    if isinstance(result.output, dict):
                        _card_data = result.output.get("card_data")
                    yield tool_result_event(
                        session_id, tc.tool_name, tc.id,
                        result.status.value, result.duration_ms,
                        output_preview, turn,
                        card_data=_card_data,
                    )

                yield turn_completed(session_id, turn, "tool_use")
                for mw in self._middleware:
                    await mw.on_turn_complete(turn, context)

                # ── Working memory stop signals ────────────────────────
                # Tools like task_complete / session_complete set flags in
                # working memory to signal "I'm done — stop the loop."
                # Without this, the engine runs another turn and the LLM
                # re-generates its report text, causing duplication.
                if working_memory.get("_task_complete") or working_memory.get("_session_complete"):
                    log.info("stop_signal_detected", session_id=session_id, turn=turn,
                             task_complete=bool(working_memory.get("_task_complete")),
                             session_complete=bool(working_memory.get("_session_complete")))
                    break  # Exit loop → finalize session

                # ── Context health monitoring ──────────────────────────
                health_config = agent_def.metadata.get("context_health")
                if isinstance(health_config, ContextHealthConfig):
                    messages_count = len(await self._repo.get_messages(session_id))
                    health = assess_health(
                        total_usage.input_tokens, messages_count,
                        health_config, already_warned=context.metadata.get(
                            "_health_warned", False),
                    )

                    if health.should_hard_stop:
                        # Force one final LLM call with no tools
                        log.warning("context_hard_stop",
                                    session_id=session_id, score=health.health_score)
                        stop_msg = AgentMessage(
                            session_id=session_id,
                            role=MessageRole.USER,
                            content=CONTEXT_HARD_STOP_MESSAGE,
                        )
                        await self._repo.add_message(stop_msg)
                        break  # Exit loop → finalize

                    if health.should_compact and not context.metadata.get(
                        "_compacted", False
                    ):
                        # Trigger LLM-based summarization of old messages
                        context.metadata["_compacted"] = True
                        log.info("context_compaction_triggered",
                                 session_id=session_id, score=health.health_score)
                        system_prompt = build_system_prompt(
                            agent_def, skills=skills,
                            working_memory_context=working_memory.get_context_for_llm(),
                            variables=variables or {},
                            objective=objective,
                            plan=working_memory.get("_plan") or [],
                            turn_number=turn,
                            max_turns=agent_def.max_turns,
                        )
                        await self._context_manager.build_context(
                            session_id, system_prompt, agent_def.model,
                            llm=self._llm,
                        )

                    if health.should_warn_agent:
                        context.metadata["_health_warned"] = True
                        warn_msg = AgentMessage(
                            session_id=session_id,
                            role=MessageRole.USER,
                            content=CONTEXT_WARNING_MESSAGE,
                        )
                        await self._repo.add_message(warn_msg)

            # Finalize session — whether we broke out naturally or hit max_turns
            # Always treat as COMPLETED. The agent did useful work either way.
            final_output = llm_response.content or ""
            if not final_output:
                msgs = await self._repo.get_messages(session_id)
                # Try assistant text first
                for m in reversed(msgs):
                    if m.role == MessageRole.ASSISTANT and m.content:
                        text = m.text() if isinstance(m.content, list) else m.content
                        if text and text.strip():
                            final_output = text
                            break
                # Fall back to tool results
                if not final_output:
                    tool_outputs = []
                    for m in reversed(msgs):
                        if m.role == MessageRole.TOOL_RESULT and m.content:
                            c = m.content if isinstance(m.content, str) else str(m.content)
                            if c.strip() and len(c.strip()) > 20:
                                tool_outputs.append(c.strip()[:1000])
                                if len(tool_outputs) >= 3:
                                    break
                    if tool_outputs:
                        tool_outputs.reverse()
                        final_output = "\n\n".join(tool_outputs)

            at_max = turn >= agent_def.max_turns
            if at_max:
                log.info("max_turns_reached", session_id=session_id, turns=turn,
                         max_turns=agent_def.max_turns)

            await self._repo.update_session(
                session_id,
                status=SessionStatus.COMPLETED,
                completed_at=datetime.now(UTC),
                turn_count=turn,
                total_input_tokens=total_usage.input_tokens,
                total_output_tokens=total_usage.output_tokens,
            )

            yield session_completed(
                session_id, turn,
                total_usage.input_tokens, total_usage.output_tokens,
                final_output,
            )

        except Exception as e:
            log.exception("engine_error", session_id=session_id)
            await self._repo.update_session(
                session_id, status=SessionStatus.FAILED, error=str(e)[:2000],
            )
            yield error_event(session_id, "engine_error", str(e)[:1000], turn)
            for mw in self._middleware:
                await mw.on_error(e, context)

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    async def _call_llm_with_retry(
        self,
        session_id: str,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        agent_def: AgentDefinition,
        system_prompt: str,
        turn: int,
        on_stream_event: Any = None,
    ) -> tuple[str, list[ToolCall], LLMUsage, StopReason]:
        """
        Call LLM with automatic retry on transient errors + provider fallback.

        Retries on:
        - Retryable LLMError (timeout, connection): exponential backoff
        - LLMRateLimitError: wait retry_after_seconds
        - ContextOverflowError: compact context, retry once

        Fallback chain: if all retries on primary model fail and
        fallback_models is configured, tries each fallback in order.

        Returns (full_text, tool_calls, usage, stop_reason).
        Raises the original error if all retries AND fallbacks exhausted.
        """
        # Build model chain: primary + fallbacks
        models_to_try = [agent_def.model] + list(agent_def.fallback_models or [])
        last_error: Exception | None = None

        for model_idx, current_model in enumerate(models_to_try):
            if model_idx > 0:
                log.warning("llm_fallback",
                            session_id=session_id,
                            failed_model=models_to_try[model_idx - 1],
                            fallback_model=current_model,
                            attempt=model_idx)

            try:
                return await self._call_single_model(
                    session_id, messages, tools, agent_def, system_prompt,
                    turn, current_model, on_stream_event=on_stream_event,
                )
            except LLMError as e:
                last_error = e
                if model_idx < len(models_to_try) - 1:
                    continue  # Try next fallback
                raise  # No more fallbacks

        # Should not reach here
        raise last_error or LLMError("All models exhausted", provider="", model=agent_def.model)

    async def _call_single_model(
        self,
        session_id: str,
        messages: list[AgentMessage],
        tools: list[ToolSchema],
        agent_def: AgentDefinition,
        system_prompt: str,
        turn: int,
        model: str,
        on_stream_event: Any = None,
    ) -> tuple[str, list[ToolCall], LLMUsage, StopReason]:
        """Call a single model with retry logic.

        Args:
            on_stream_event: Optional async callback(StreamEvent) invoked for
                each text_delta / thinking chunk during streaming. This enables
                real-time token streaming to SSE clients while the engine still
                accumulates the full response for persistence.
        """
        max_retries = agent_def.max_llm_retries
        context_compacted = False

        timeout = agent_def.llm_timeout_seconds

        for attempt in range(max_retries + 1):
            try:
                response = self._llm.stream_with_tools(
                    messages=messages,
                    tools=tools,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=agent_def.temperature,
                    max_tokens=agent_def.max_tokens_per_turn,
                )

                # Collect streamed response with timeout
                full_text = ""
                tool_calls: list[ToolCall] = []
                usage = LLMUsage()
                stop_reason = StopReason.END_TURN

                async def _collect_stream():
                    nonlocal full_text, usage, stop_reason
                    async for chunk in response:
                        if chunk.type == "text_delta" and chunk.text:
                            full_text += chunk.text
                            # Emit real-time text streaming event
                            if on_stream_event:
                                await on_stream_event(
                                    text_delta(session_id, chunk.text, turn)
                                )
                        elif chunk.type == "thinking_delta" and chunk.text:
                            # Emit real-time thinking streaming event
                            if on_stream_event:
                                await on_stream_event(
                                    thinking_delta(session_id, chunk.text, turn)
                                )
                        elif chunk.type == "tool_call_end" and chunk.tool_call:
                            tool_calls.append(chunk.tool_call)
                            # Emit tool_call event for SSE buffering
                            if on_stream_event and chunk.tool_call:
                                await on_stream_event(
                                    tool_call_event(
                                        session_id,
                                        chunk.tool_call.tool_name,
                                        chunk.tool_call.id,
                                        chunk.tool_call.arguments,
                                        turn,
                                    )
                                )
                        elif chunk.type == "usage":
                            if chunk.usage:
                                usage = chunk.usage
                            if chunk.stop_reason:
                                stop_reason = chunk.stop_reason
                        elif chunk.type == "complete":
                            if chunk.text:
                                full_text = chunk.text
                            if chunk.usage:
                                usage = chunk.usage
                            if chunk.stop_reason:
                                stop_reason = chunk.stop_reason

                try:
                    await asyncio.wait_for(_collect_stream(), timeout=timeout)
                except TimeoutError:
                    raise LLMError(
                        f"LLM call timed out after {timeout}s",
                        provider="", model=agent_def.model, retryable=True,
                    )

                return full_text, tool_calls, usage, stop_reason

            except LLMRateLimitError as e:
                if attempt >= max_retries:
                    raise
                wait = e.retry_after_seconds or min(2 ** attempt, 30)
                log.warning("llm_rate_limit_retry",
                            session_id=session_id, attempt=attempt + 1,
                            max_retries=max_retries, wait_seconds=wait)
                await asyncio.sleep(wait)

            except ContextOverflowError:
                if context_compacted or attempt >= max_retries:
                    raise
                log.warning("context_overflow_retry",
                            session_id=session_id, attempt=attempt + 1)
                # Compact context and retry once
                messages = await self._context_manager.build_context(
                    session_id, system_prompt, agent_def.model, llm=self._llm,
                )
                context_compacted = True

            except LLMError as e:
                if not e.retryable or attempt >= max_retries:
                    raise
                wait = min(2 ** attempt, 30)
                log.warning("llm_error_retry",
                            session_id=session_id, attempt=attempt + 1,
                            max_retries=max_retries, wait_seconds=wait,
                            error=str(e)[:200])
                await asyncio.sleep(wait)

        # Should not reach here, but just in case
        raise LLMError("All retry attempts exhausted", provider="", model=model)

    async def _load_knowledge_index(self, agent_id: str) -> list[dict]:
        """Load the knowledge document listing for prompt injection."""
        try:
            all_memories = await self._repo.list_memories(agent_id, memory_type="document")
            return [
                {
                    "name": m["key"].removeprefix("doc:"),
                    "size": len(str(m.get("value", ""))),
                }
                for m in all_memories
                if m.get("key", "").startswith("doc:")
            ]
        except Exception:
            return []

    async def _load_skill_index(self, agent_id: str) -> list[dict]:
        """Load the skill listing for prompt injection."""
        try:
            all_memories = await self._repo.list_memories(agent_id, memory_type="skill")
            result = []
            for m in all_memories:
                if m.get("key", "").startswith("skill:"):
                    val = m.get("value", {})
                    desc = val.get("description", "") if isinstance(val, dict) else ""
                    result.append({
                        "name": m["key"].removeprefix("skill:"),
                        "description": desc,
                    })
            return result
        except Exception:
            return []

    async def _load_objective(self, agent_id: str) -> str | None:
        """Load the agent's objective (full content) for prompt injection."""
        try:
            value = await self._repo.get_memory(agent_id, "objective")
            if value is None:
                return None
            return value if isinstance(value, str) else str(value)
        except Exception:
            return None

    async def _ensure_session(
        self,
        session_id: str,
        agent_def: AgentDefinition,
        metadata: dict | None,
    ) -> AgentSession:
        """Load existing session or create new one."""
        session = await self._repo.get_session(session_id)
        if session:
            return session

        safe_metadata = _json_safe(metadata) if metadata else {}

        session = AgentSession(
            id=session_id,
            agent_id=agent_def.id,
            config=agent_def.model_dump(exclude={"sub_agents"}),
            metadata=safe_metadata,
        )
        return await self._repo.create_session(session)
