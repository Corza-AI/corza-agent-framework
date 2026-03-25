# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-24

### Added
- **Parallel sub-agent dispatch**: `manage_agent(action="spawn_parallel")` runs multiple sub-agents concurrently via `asyncio.gather`. Configurable via `max_parallel_agents` on `AgentDefinition` (default: 5).
- **Nuclear stop**: `orchestrator.cancel(session_id)` cancels a session and ALL child sessions. HTTP: `POST /sessions/{id}/cancel`. Running loops exit at the top of the next turn.
- **Session cancellation**: `engine.cancel(session_id, cascade=True)` with cascade to child sessions. New `get_child_sessions()` on all repository backends.
- **SSE production streaming**: Router now uses `sse_response()` from `streaming/sse.py` with proper queue-based heartbeat (fires during long LLM calls), Last-Event-ID reconnection, and client disconnect detection.
- **Auth guards**: `POST /tools` and `POST /agents` return 403 by default. Opt-in with `admin_only=False` on `create_agent_router()`.
- **Code execution safety gate**: `execute_code_tool()` requires `CORZA_ALLOW_CODE_EXECUTION=true` env var. Disabled by default.
- **Turn-budget awareness**: System prompt shows `Turn N of M` every turn. Prompts instruct agents to skip to SYNTHESIZE/DELIVER when <3 turns remain.
- **Sub-agent restrictions**: Task agents can only use `manage_agent(action="report")`. All other actions (spawn, message, list) return errors.
- **User/tenant auth context**: `ExecutionContext` now has typed `user_id` and `tenant_id` fields, wired from session metadata.
- **Objective management**: `manage_objective` tool for persistent mission/purpose (auto-loaded into prompt every turn).
- **Manual context compaction**: `manage_context` tool sets a flag → engine triggers LLM summarization on the next turn.
- **Scheduler**: DB-backed cron/one-time/event scheduling for automated agent runs.
- **235 tests** across 26 test files covering engine, orchestrator, delegation, SSE, security, router, prompts, persistence, middleware, scheduler.

### Changed
- **OPTIMIZE before DELIVER**: Task agent phases reordered (PREPARE → PLAN → INVESTIGATE → OPTIMIZE → REPORT). Both prompts warn: "Complete all persist calls BEFORE your final response."
- **Prompts enhanced**: spawn_parallel documented, tool failure retry guidance, interim vs final report labels, "only use listed tools" guardrail, "handle tool failures" principle.
- **SubAgentRunner encapsulation**: Uses `engine.repository` property instead of `engine._repo`.
- **Dead code removed**: Router's inline `_stream_events()` with broken heartbeat replaced by `sse_response()`.

### Fixed
- SSE heartbeat now fires during long LLM calls (was dead code in router).
- `manage_context` tool actually triggers compaction (was a no-op).
- Security section and troubleshooting section added to README.

## [0.2.0] - 2026-03-24

### Added
- **Multi-backend persistence**: `InMemoryRepository`, `SQLiteRepository` alongside existing PostgreSQL. Use `create_repository("memory"|"sqlite"|"postgres")`.
- **LLM retry & error recovery**: Automatic retry on rate limits, timeouts, and context overflow. Configurable via `max_llm_retries` on `AgentDefinition`.
- **Turn-level recovery**: Failed LLM turns set session to `WAITING_INPUT` instead of `FAILED`, allowing resume on next message.
- **AgentService**: Framework-agnostic service layer (`api/service.py`). No web framework imports — use with FastAPI, Flask, Django, or CLI.
- **Programmatic skills**: `SkillsManager.from_function()`, `from_url()`, `from_database()` for creating skills from code, URLs, or database callbacks.
- **Dynamic knowledge sources**: `knowledge` field now accepts `str` (file paths), `dict` (inline), or `Callable` (dynamic).
- **Runtime skill injection**: `ExecutionContext.add_skill()` / `remove_skill()` for mid-session skill changes.
- **BaseRepository ABC**: All persistence backends implement a common interface.
- **Repository factory**: `create_repository()` function for one-line backend selection.
- **Public Orchestrator properties**: `.repo`, `.tools`, `.middleware`, `.skills` accessors.
- `get_audit_log()` method on all repository backends.

### Changed
- **Model is now required**: `AgentDefinition.model` defaults to `""` — you must explicitly specify a model string. Clear error message if omitted.
- **PostgreSQL is optional**: `asyncpg` and `sqlalchemy` moved from core dependencies to the `postgres` extra. Core framework has zero database dependencies.
- **FastAPI router decoupled**: Router is now a thin adapter over `AgentService`. No private attribute access on Orchestrator.
- `Repository` renamed to `PostgresRepository` (backward-compatible alias `Repository` still works).

### Fixed
- Examples now use correct API (`create_repository`, explicit model, proper event access).
- Working memory tool examples use sync `.store()` / `.get()` (not async).

## [0.1.0] - 2025-12-01

### Added
- Initial release: ReAct engine, multi-agent orchestration, tool system, skills, middleware, streaming, PostgreSQL persistence.
