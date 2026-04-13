"""
Corza Agent Framework — Orchestrator

The brain agent. Wraps the AgentEngine with sub-agent delegation,
wiring the manage_agent built-in tool to actual sub-agent execution.

Usage:
    orchestrator = Orchestrator(llm, tool_registry, repository, ...)
    orchestrator.register_sub_agent("analyst", analyst_def)
    orchestrator.register_sub_agent("researcher", researcher_def)

    async for event in orchestrator.run(session_id, "Investigate case X", brain_def):
        handle(event)
"""

import json
from collections.abc import AsyncIterator

import structlog

from corza_agents.core.engine import AgentEngine
from corza_agents.core.llm import AgentLLM
from corza_agents.core.types import (
    AgentDefinition,
    ExecutionContext,
    SubAgentResult,
    _uuid,
)
from corza_agents.middleware.audit import AuditMiddleware
from corza_agents.middleware.base import BaseMiddleware
from corza_agents.middleware.token_tracking import TokenTrackingMiddleware
from corza_agents.orchestrator.sub_agent import SubAgentRunner
from corza_agents.persistence.base import BaseRepository
from corza_agents.prompts.templates import ORCHESTRATOR_SYSTEM_PROMPT, TASK_AGENT_SYSTEM_PROMPT
from corza_agents.skills.manager import SkillsManager
from corza_agents.streaming.events import (
    StreamEvent,
)
from corza_agents.tools.builtin import BUILTIN_TOOLS
from corza_agents.tools.registry import ToolRegistry

log = structlog.get_logger("corza_agents.orchestrator")


class Orchestrator:
    """
    The brain agent. Plans, delegates to sub-agents, synthesizes results.

    This is the top-level entry point for running agents in this system.
    It manages:
    - The core AgentEngine (ReAct loop)
    - Sub-agent registration and delegation
    - Built-in tools (manage_agent, manage_knowledge, manage_plan, etc.)
    - Default middleware (audit, token tracking)
    - Skill resolution

    The orchestrator wires the manage_agent tool to the SubAgentRunner,
    so when the LLM calls manage_agent, it actually spawns a sub-agent.
    """

    def __init__(
        self,
        llm: AgentLLM,
        tool_registry: ToolRegistry,
        repository: BaseRepository,
        skills_manager: SkillsManager | None = None,
        middleware: list[BaseMiddleware] | None = None,
        auto_register_builtins: bool = True,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._repo = repository
        self._skills = skills_manager or SkillsManager()
        self._sub_agents: dict[str, AgentDefinition] = {}

        # Build middleware stack (order matters)
        mw_stack: list[BaseMiddleware] = []
        mw_stack.append(AuditMiddleware(repository))
        mw_stack.append(TokenTrackingMiddleware(repository))
        if middleware:
            mw_stack.extend(middleware)
        self._middleware = mw_stack

        # Create the engine
        self._engine = AgentEngine(
            llm=llm,
            tool_registry=tool_registry,
            repository=repository,
            skills_manager=self._skills,
            middleware=self._middleware,
        )

        # Sub-agent runner
        self._sub_agent_runner = SubAgentRunner(self._engine)

        # Register built-in tools (skip any already registered by the caller
        # to allow applications to provide their own implementations)
        if auto_register_builtins:
            for builtin_fn in BUILTIN_TOOLS:
                if hasattr(builtin_fn, "tool_definition"):
                    td = builtin_fn.tool_definition
                    if not self._tools.has(td.name):
                        self._tools.register(td)

        # Wire the manage_agent tool
        self._wire_delegation_tool()

    # ══════════════════════════════════════════════════════════════════
    # Sub-Agent Registration
    # ══════════════════════════════════════════════════════════════════

    def register_sub_agent(self, name: str, agent_def: AgentDefinition) -> None:
        """
        Register a sub-agent that can be invoked via manage_agent.

        If the sub-agent has no system prompt, the framework injects
        TASK_AGENT_SYSTEM_PROMPT — proven investigation principles
        (depth over breadth, schema first, memory first, record immediately).
        """
        if not agent_def.system_prompt:
            agent_def = agent_def.model_copy(
                update={
                    "system_prompt": TASK_AGENT_SYSTEM_PROMPT,
                }
            )
        self._sub_agents[name] = agent_def
        log.info("sub_agent_registered", name=name, model=agent_def.model, tools=agent_def.tools)

    def register_sub_agents(self, agents: dict[str, AgentDefinition]) -> None:
        for name, agent_def in agents.items():
            self.register_sub_agent(name, agent_def)

    @property
    def sub_agents(self) -> dict[str, AgentDefinition]:
        return dict(self._sub_agents)

    @property
    def repo(self) -> BaseRepository:
        """Public accessor for the persistence repository."""
        return self._repo

    @property
    def tools(self) -> ToolRegistry:
        """Public accessor for the tool registry."""
        return self._tools

    @property
    def middleware(self) -> list[BaseMiddleware]:
        """Public accessor for the middleware stack."""
        return list(self._middleware)

    @property
    def skills(self) -> SkillsManager:
        """Public accessor for the skills manager."""
        return self._skills

    # ══════════════════════════════════════════════════════════════════
    # Main Entry Point
    # ══════════════════════════════════════════════════════════════════

    async def run(
        self,
        session_id: str,
        user_message: str,
        agent_def: AgentDefinition,
        metadata: dict | None = None,
        variables: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Run the orchestrator agent loop.

        Behaves like AgentEngine.run() but with sub-agent delegation wired up.
        """
        # Add sub-agent definitions to the agent_def for prompt generation
        if self._sub_agents:
            agent_def = agent_def.model_copy(
                update={
                    "sub_agents": self._sub_agents,
                }
            )

        # If the brain agent has sub-agents but no system prompt, inject the
        # orchestrator prompt so it knows how to plan, delegate, and synthesize.
        if self._sub_agents and not agent_def.system_prompt:
            agent_def = agent_def.model_copy(
                update={
                    "system_prompt": ORCHESTRATOR_SYSTEM_PROMPT,
                }
            )

        # Augment system prompt with available sub-agents info
        if self._sub_agents:
            sub_agent_info = self._build_sub_agent_prompt()
            current_prompt = agent_def.system_prompt or ""
            agent_def = agent_def.model_copy(
                update={
                    "system_prompt": current_prompt + "\n\n" + sub_agent_info,
                }
            )

        # Ensure manage_agent is in the tool list
        if "manage_agent" not in agent_def.tools and self._sub_agents:
            agent_def = agent_def.model_copy(
                update={
                    "tools": agent_def.tools + ["manage_agent"],
                }
            )

        # Pass max_parallel_agents into metadata so the handler can enforce it
        run_metadata = dict(metadata or {})
        run_metadata["max_parallel_agents"] = agent_def.max_parallel_agents

        async for event in self._engine.run(
            session_id=session_id,
            user_message=user_message,
            agent_def=agent_def,
            metadata=run_metadata,
            variables=variables,
        ):
            yield event

    async def run_simple(
        self,
        user_message: str,
        agent_def: AgentDefinition,
        metadata: dict | None = None,
    ) -> str:
        """
        Simple interface: run agent to completion and return final text.
        Generates a session ID automatically.
        """
        session_id = _uuid()
        final_output = ""

        async for event in self.run(session_id, user_message, agent_def, metadata):
            if event.type.value == "session.completed":
                final_output = event.data.get("final_output", "")
            elif event.type.value == "error":
                raise Exception(event.data.get("message", "Agent error"))

        return final_output

    # ══════════════════════════════════════════════════════════════════
    # Initialization
    # ══════════════════════════════════════════════════════════════════

    async def initialize(self) -> None:
        """Initialize DB tables. Call once on startup."""
        await self._repo.initialize()
        log.info(
            "orchestrator_initialized",
            tools=list(self._tools.tools.keys()),
            sub_agents=list(self._sub_agents.keys()),
            middleware=[m.name for m in self._middleware],
        )

    async def cancel(self, session_id: str) -> int:
        """
        Nuclear stop — cancel a session and ALL its child sessions.

        Stops the orchestrator's ReAct loop AND every sub-agent it spawned.
        Sessions are set to CANCELLED status immediately. Running loops
        detect this at the top of the next turn and exit.

        Args:
            session_id: The orchestrator session to cancel.

        Returns:
            Number of sessions cancelled (parent + children).
        """
        return await self._engine.cancel(session_id, cascade=True)

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self._repo.close()

    # ══════════════════════════════════════════════════════════════════
    # Internal: Wire manage_agent to actual sub-agent execution
    # ══════════════════════════════════════════════════════════════════

    def _wire_delegation_tool(self) -> None:
        """
        Replace the placeholder manage_agent handler with full lifecycle
        management: spawn, message, status, list.
        """
        tool_def = self._tools.get("manage_agent")
        if not tool_def:
            return

        runner = self._sub_agent_runner
        sub_agents = self._sub_agents
        repo = self._repo

        async def _manage_agent(
            action: str,
            agent_name: str = "",
            task: str = "",
            tasks: str = "",
            session_id: str = "",
            context_data: str = "",
            content: str = "",
            plan_item_id: str = "",
            ctx: ExecutionContext = None,
        ) -> dict:
            # Task agents (sub-agents) can ONLY report back — they cannot
            # spawn, message, or inspect other agents.
            is_sub_agent = bool(
                ctx
                and ctx.parent_session_id
                or (ctx and ctx.metadata and ctx.metadata.get("parent_session_id"))
            )
            if is_sub_agent and action != "report":
                return {
                    "status": "error",
                    "message": (
                        f"As a task agent, you can only use action='report' to send "
                        f"findings to the orchestrator. Action '{action}' is not available. "
                        f"Use: manage_agent(action='report', content='your findings...')"
                    ),
                }

            if action == "list":
                return await _list_agents(ctx)
            elif action == "spawn":
                return await _spawn(agent_name, task, context_data, ctx, plan_item_id)
            elif action == "spawn_parallel":
                return await _spawn_parallel(tasks, ctx)
            elif action == "message":
                return await _message(session_id, task, ctx)
            elif action == "report":
                return await _report(content, ctx)
            elif action == "status":
                return await _status(session_id)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action '{action}'. Use: spawn, spawn_parallel, message, report, status, list.",
                }

        async def _list_agents(ctx):
            available = {
                name: {"description": d.description, "model": d.model, "tools": d.tools}
                for name, d in sub_agents.items()
            }
            return {"status": "success", "agents": available, "count": len(available)}

        async def _spawn(agent_name, task, context_data, ctx, plan_item_id=""):
            if not agent_name:
                return {"status": "error", "message": "Provide 'agent_name' to spawn."}
            if not task:
                return {"status": "error", "message": "Provide 'task' for the sub-agent."}
            if agent_name not in sub_agents:
                available = list(sub_agents.keys())
                return {
                    "status": "error",
                    "message": f"Sub-agent '{agent_name}' not found. Available: {available}",
                }

            agent_def = sub_agents[agent_name]
            context_dict = None
            if context_data:
                try:
                    context_dict = json.loads(context_data)
                except (json.JSONDecodeError, TypeError):
                    context_dict = {"context": context_data}

            parent_sid = ctx.session_id if ctx else ""

            on_spawn = ctx.metadata.get("on_subagent_spawn") if ctx and ctx.metadata else None
            on_event = ctx.metadata.get("on_subagent_event") if ctx and ctx.metadata else None

            # ── External dispatch mode ──────────────────────────────────
            # If on_subagent_spawn returns a dict, the caller has FULLY
            # handled the dispatch (e.g. created a background task with its
            # own streaming, DB persistence, etc.). We return the result
            # directly — do NOT run the sub-agent inline via runner.run(),
            # which would execute it a second time.
            if on_spawn:
                spawn_result = await on_spawn(agent_name, task, parent_sid)
                if isinstance(spawn_result, dict):
                    log.info(
                        "agent_dispatched_externally",
                        agent_name=agent_name,
                        task_id=spawn_result.get("task_id", "?"),
                    )
                    return {
                        "status": "dispatched",
                        "message": (
                            f"Task '{spawn_result.get('title', task[:80])}' has been dispatched "
                            f"and is running in the background (task_id: {spawn_result.get('task_id', '?')}). "
                            f"Results will appear when the task completes. "
                            f"Do NOT re-spawn this task — it is already running. "
                            f"Move on to your next plan item or wait for results."
                        ),
                        **spawn_result,
                    }
                # String return = session_id for inline execution (original behavior)
                child_session_id = spawn_result
            else:
                child_session_id = None

            # ── Inline execution mode ───────────────────────────────────
            # No external dispatch — run the sub-agent inline via the
            # framework's SubAgentRunner. Blocks until the sub-agent completes.
            event_callback = None
            if on_event and child_session_id:
                cid = child_session_id

                async def event_callback(event):
                    await on_event(cid, event)

            child_metadata = {}
            if ctx and ctx.metadata:
                for key in (
                    "workspace_path",
                    "llm",
                    "model",
                    "connection_id",
                    "table_name",
                    "database",
                    "data_source",
                    "schema_context",
                    "dialect",
                    "tables",
                    "schema_hint",
                ):
                    if key in ctx.metadata:
                        child_metadata[key] = ctx.metadata[key]

            log.info(
                "child_metadata_built",
                parent_session=parent_sid[:8] if parent_sid else "?",
                connection_id=child_metadata.get("connection_id", "MISSING"),
                keys=list(child_metadata.keys()),
                parent_metadata_keys=list(ctx.metadata.keys()) if ctx and ctx.metadata else [],
            )

            # Auto-mark plan item in_progress before spawn
            if plan_item_id and ctx and ctx.working_memory:
                plan = ctx.working_memory.get("_plan") or []
                for item in plan:
                    if item.get("id") == plan_item_id:
                        item["status"] = "in_progress"
                        break
                ctx.working_memory.store("_plan", plan)

            result: SubAgentResult = await runner.run(
                task=task,
                agent_def=agent_def,
                parent_session_id=parent_sid,
                context_data=context_dict,
                metadata=child_metadata,
                on_event=event_callback,
            )

            on_complete = ctx.metadata.get("on_subagent_complete") if ctx and ctx.metadata else None
            completed_child_id = child_session_id or result.child_session_id
            report_id = None
            if on_complete and completed_child_id:
                cb_result = await on_complete(completed_child_id, result.status.value, result.output)
                if isinstance(cb_result, dict):
                    report_id = cb_result.get("report_id")

            # Auto-mark plan item done/blocked based on result
            if plan_item_id and ctx and ctx.working_memory:
                plan = ctx.working_memory.get("_plan") or []
                new_status = "done" if result.status.value == "success" else "blocked"
                for item in plan:
                    if item.get("id") == plan_item_id:
                        item["status"] = new_status
                        break
                ctx.working_memory.store("_plan", plan)

            log.info(
                "agent_spawned",
                agent_name=agent_name,
                child_session_id=result.child_session_id,
                status=result.status.value,
                turns=result.turns_used,
            )

            return {
                "status": result.status.value,
                "report": result.output or "",
                "report_id": report_id,
                "error": result.error,
            }

        async def _spawn_parallel(tasks_json, ctx):
            """Spawn multiple sub-agents concurrently and return all results."""
            if not tasks_json:
                return {
                    "status": "error",
                    "message": (
                        "Provide 'tasks' as a JSON array of objects: "
                        '[{"agent_name": "researcher", "task": "Investigate X"}, ...]'
                    ),
                }

            try:
                task_list = json.loads(tasks_json) if isinstance(tasks_json, str) else tasks_json
            except (json.JSONDecodeError, TypeError):
                return {
                    "status": "error",
                    "message": "Invalid 'tasks' JSON. Expected array of {agent_name, task} objects.",
                }

            if not isinstance(task_list, list) or not task_list:
                return {"status": "error", "message": "Provide a non-empty array of task objects."}

            # Enforce configurable concurrency limit
            max_parallel = 5  # default
            if ctx and ctx.metadata:
                max_parallel = ctx.metadata.get("max_parallel_agents", max_parallel)
            if len(task_list) > max_parallel:
                return {
                    "status": "error",
                    "message": (
                        f"Too many parallel tasks ({len(task_list)}). "
                        f"Maximum is {max_parallel}. Split into batches."
                    ),
                }

            # Validate all tasks before spawning any
            for i, t in enumerate(task_list):
                if not isinstance(t, dict):
                    return {"status": "error", "message": f"Task {i} is not an object."}
                if not t.get("agent_name"):
                    return {"status": "error", "message": f"Task {i} missing 'agent_name'."}
                if not t.get("task"):
                    return {"status": "error", "message": f"Task {i} missing 'task'."}
                if t["agent_name"] not in sub_agents:
                    available = list(sub_agents.keys())
                    return {
                        "status": "error",
                        "message": (
                            f"Task {i}: agent '{t['agent_name']}' not found. Available: {available}"
                        ),
                    }

            # Spawn all concurrently
            import asyncio as _asyncio

            async def _run_one(t):
                return await _spawn(
                    t["agent_name"],
                    t["task"],
                    t.get("context_data", ""),
                    ctx,
                )

            results = await _asyncio.gather(
                *[_run_one(t) for t in task_list],
                return_exceptions=True,
            )

            # Format results — cap individual outputs to prevent context overflow
            max_output_chars = 4000  # Per agent — enough for key findings + chart IDs
            formatted = []
            for i, (t, r) in enumerate(zip(task_list, results)):
                if isinstance(r, Exception):
                    formatted.append(
                        {
                            "agent_name": t["agent_name"],
                            "task": t["task"][:200],
                            "status": "error",
                            "error": str(r)[:500],
                        }
                    )
                else:
                    r["task"] = t["task"][:200]
                    # Truncate long outputs but preserve chart tokens
                    if isinstance(r.get("output"), str) and len(r["output"]) > max_output_chars:
                        output = r["output"]
                        # Keep the first chunk + any chart tokens from the rest
                        import re

                        chart_tokens = re.findall(r"\{\{chart:[^}]+\}\}", output[max_output_chars:])
                        r["output"] = output[:max_output_chars] + (
                            f"\n\n... (truncated — {len(output)} chars total)"
                            + (
                                "\n\nAdditional charts: " + " ".join(chart_tokens)
                                if chart_tokens
                                else ""
                            )
                        )
                    formatted.append(r)

            succeeded = sum(
                1 for r in formatted if isinstance(r, dict) and r.get("status") == "success"
            )
            log.info(
                "parallel_spawn_complete",
                total=len(task_list),
                succeeded=succeeded,
                failed=len(task_list) - succeeded,
            )

            return {
                "status": "success",
                "action": "spawn_parallel",
                "total": len(task_list),
                "succeeded": succeeded,
                "results": formatted,
            }

        async def _message(child_session_id, message, ctx):
            if not child_session_id:
                return {"status": "error", "message": "Provide 'session_id' of the sub-agent."}
            if not message:
                return {"status": "error", "message": "Provide 'task' as the follow-up message."}

            # Look up the child session to find its agent definition
            child_session = await repo.get_session(child_session_id)
            if not child_session:
                return {
                    "status": "error",
                    "message": f"Sub-agent session '{child_session_id}' not found.",
                }

            # Find the matching agent definition
            agent_def = None
            for name, defn in sub_agents.items():
                if defn.id == child_session.agent_id:
                    agent_def = defn
                    break

            if not agent_def:
                # Fallback: use the first sub-agent definition
                agent_def = next(iter(sub_agents.values()), None)

            if not agent_def:
                return {"status": "error", "message": "No agent definition found for this session."}

            on_event = ctx.metadata.get("on_subagent_event") if ctx and ctx.metadata else None
            event_callback = None
            if on_event:
                cid = child_session_id

                async def event_callback(event):
                    await on_event(cid, event)

            result: SubAgentResult = await runner.send_message(
                child_session_id=child_session_id,
                message=message,
                agent_def=agent_def,
                on_event=event_callback,
            )

            log.info(
                "agent_messaged",
                session_id=child_session_id,
                status=result.status.value,
                turns=result.turns_used,
            )

            return {
                "status": result.status.value,
                "output": result.output,
                "session_id": child_session_id,
                "turns_used": result.turns_used,
                "tokens_used": result.tokens_used.total_tokens,
                "error": result.error,
            }

        async def _report(content, ctx):
            """Sub-agent reports findings back to the parent orchestrator."""
            if not content:
                return {"status": "error", "message": "Provide 'content' with your findings."}
            if not ctx:
                return {"status": "error", "message": "No execution context available."}

            parent_sid = ctx.metadata.get("parent_session_id", "")
            if not parent_sid:
                # Not a sub-agent — store locally
                if ctx.working_memory:
                    ctx.working_memory.add_finding(
                        description=content[:500],
                        category="report",
                        evidence={"full_content": content},
                    )
                return {
                    "status": "success",
                    "stored": "local",
                    "message": "Report stored in local working memory (no parent session).",
                }

            # Store the report as a message in the PARENT session
            from corza_agents.core.types import AgentMessage, MessageRole

            report_msg = AgentMessage(
                session_id=parent_sid,
                role=MessageRole.USER,
                content=(
                    f"[REPORT from sub-agent '{ctx.agent_name}' "
                    f"(session: {ctx.session_id})]\n\n{content}"
                ),
            )
            await repo.add_message(report_msg)

            # Also persist as a document for cross-session access
            report_key = f"doc:report-{ctx.agent_name}-{ctx.session_id[:8]}"
            await repo.set_memory(
                ctx.agent_id,
                report_key,
                content,
                memory_type="document",
                session_id=ctx.session_id,
            )

            log.info(
                "agent_report",
                child_session_id=ctx.session_id,
                parent_session_id=parent_sid,
                agent_name=ctx.agent_name,
                report_length=len(content),
            )

            return {
                "status": "success",
                "stored": "parent",
                "parent_session_id": parent_sid,
                "message": f"Report sent to orchestrator (parent session {parent_sid[:8]}...).",
            }

        async def _status(child_session_id):
            if not child_session_id:
                return {"status": "error", "message": "Provide 'session_id' to check."}
            session = await repo.get_session(child_session_id)
            if not session:
                return {"status": "error", "message": f"Session '{child_session_id}' not found."}
            return {
                "status": "success",
                "session_id": child_session_id,
                "agent_status": session.status.value,
                "turn_count": session.turn_count,
                "tokens": session.total_input_tokens + session.total_output_tokens,
                "error": session.error,
            }

        tool_def = tool_def.model_copy(update={"handler": _manage_agent})
        self._tools.register(tool_def)

    def _build_sub_agent_prompt(self) -> str:
        """Build system prompt section describing available sub-agents."""
        lines = [
            "## Available Sub-Agents\n",
            "Use the `manage_agent` tool to delegate tasks to these specialized agents:\n",
        ]
        for name, agent_def in self._sub_agents.items():
            lines.append(f"### {name}")
            lines.append(f"- **Description**: {agent_def.description}")
            lines.append(f"- **Model**: {agent_def.model}")
            if agent_def.tools:
                lines.append(f"- **Tools**: {', '.join(agent_def.tools)}")
            if agent_def.skills:
                lines.append(f"- **Skills**: {', '.join(agent_def.skills)}")
            lines.append("")
        return "\n".join(lines)
