# Architecture

This document explains how the Corza Agent Framework is structured internally — the ReAct loop, component responsibilities, data flow, and extension points.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Your Application                              │
│                                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐ │
│  │ Auth Layer  │───▶│ FastAPI      │───▶│ AgentService               │ │
│  │ (yours)     │    │ Router       │    │ (framework-agnostic)       │ │
│  └─────────────┘    └──────────────┘    └──────────┬──────────────────┘ │
│                                                     │                    │
│                            ┌────────────────────────▼─────────────────┐ │
│                            │          Orchestrator                    │ │
│                            │  ┌─────────────┐  ┌─────────────────┐  │ │
│                            │  │ Brain Agent  │  │  Sub-Agents     │  │ │
│                            │  │ (delegates)  │──│  (specialized)  │  │ │
│                            │  └──────┬──────┘  └─────────────────┘  │ │
│                            └─────────┼──────────────────────────────┘ │
│                                      │                                │
│  ┌───────────────────────────────────▼──────────────────────────────┐ │
│  │                      AgentEngine (ReAct Loop)                    │ │
│  │                                                                   │ │
│  │  ┌──────────┐   ┌──────────────┐   ┌───────────────────────────┐│ │
│  │  │ AgentLLM │   │ ToolRegistry │   │ Memory                   ││ │
│  │  │ 23+      │   │ @tool deco   │   │ Working + Context + Health││ │
│  │  │ providers│   │ JSON schema  │   │                           ││ │
│  │  └──────────┘   └──────────────┘   └───────────────────────────┘│ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Middleware Pipeline                                              │ │
│  │  ContextCompression → RateLimit → Audit → TokenTracking          │ │
│  │  → Permission → LoopGuard → [Custom]                             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Persistence                                                      │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐                   │ │
│  │  │ In-Memory│  │  SQLite  │  │  PostgreSQL  │                   │ │
│  │  │ (tests)  │  │  (dev)   │  │ (production) │                   │ │
│  │  └──────────┘  └──────────┘  └──────────────┘                   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The ReAct Loop

The core of the framework is a **Reason + Act** loop with automatic retry and error recovery:

```
User message
     │
     ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  LLM Call   │ ───▶ │  Tool Exec  │ ───▶ │  LLM Call   │ ───▶ ... ───▶ Final Answer
│  (reason)   │      │  (act)      │      │  (reason)   │
└─────────────┘      └─────────────┘      └─────────────┘
      ▲                                          │
      │          persisted after each turn        │
      └──────────────────────────────────────────┘
```

**Turn lifecycle:**

1. Engine sends conversation history + tool schemas to the LLM
2. LLM responds with either text (done) or tool calls (continue)
3. Engine executes each tool call through the ToolRegistry
4. Tool results are appended to conversation history
5. Loop repeats from step 1

**Termination conditions:**
- LLM produces a final text response (no tool calls)
- `max_turns` limit reached
- Unrecoverable error after all retries exhausted

---

## Component Reference

| Component | Module | Responsibility |
|-----------|--------|---------------|
| **AgentLLM** | `core/llm.py` | Unified LLM client for 23+ providers. Handles streaming, tool schema conversion, message format normalization, and provider-specific quirks |
| **AgentEngine** | `core/engine.py` | Runs the ReAct loop. Manages session lifecycle, calls LLM with retry, executes tools, emits streaming events |
| **ToolRegistry** | `tools/registry.py` | Central tool store. Auto-generates JSON schemas from Python type hints. Dispatches execution with timeout and retry |
| **Orchestrator** | `orchestrator/orchestrator.py` | Multi-agent coordinator. Registers sub-agents, provides `manage_agent` tool to the brain, handles parallel dispatch |
| **SubAgentRunner** | `orchestrator/sub_agent.py` | Isolates sub-agent execution. Each sub-agent gets its own session, tool set, and message history |
| **SkillsManager** | `skills/manager.py` | Loads skills from files, functions, URLs, or databases. Injects skill prompts into the system message |
| **WorkingMemory** | `memory/working.py` | Per-session scratch space. Tools store/retrieve data during a run |
| **ContextManager** | `memory/context.py` | Keeps the context window under control via truncation, compression, and LLM summarization |
| **BaseMiddleware** | `middleware/base.py` | Hook interface with 6 extension points: before/after LLM, before/after tool, on turn complete, on error |
| **AgentService** | `api/service.py` | Framework-agnostic service layer. No web framework imports — works with FastAPI, Flask, Django, or CLI |
| **BaseRepository** | `persistence/base.py` | Abstract persistence interface. Implemented by InMemory, SQLite, and PostgreSQL backends |

---

## Data Flow: Message Lifecycle

```
1. HTTP Request                    2. AgentService                 3. Orchestrator
   POST /sessions/{id}/messages ──▶  service.send_message() ───▶  orchestrator.run()
                                                                        │
                                                                        ▼
4. AgentEngine                     5. AgentLLM                    6. Tool Execution
   engine.run() ─────────────────▶  llm.stream_response() ──▶    tool_registry.execute()
        │                                  │                           │
        │    ┌─────────────────────────────┘                           │
        │    │  StreamEvent(llm.text_delta)                            │
        │    │  StreamEvent(llm.tool_call)                             │
        │    ▼                                                         │
        │  SSE to client ◀──────── StreamEvent(tool.result) ◀─────────┘
        │
        ▼
7. Persistence
   repository.save_message()
   repository.save_tool_execution()
```

---

## Persistence Layer

### Database Schema

All three backends (InMemory, SQLite, PostgreSQL) store the same tables:

```
┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   af_sessions    │     │   af_messages    │     │ af_tool_executions │
├──────────────────┤     ├──────────────────┤     ├────────────────────┤
│ id (PK)          │◀───│ session_id (FK)  │     │ session_id (FK)    │
│ agent_name       │     │ role             │     │ tool_name          │
│ status           │     │ content          │     │ arguments (JSON)   │
│ user_id          │     │ tool_calls (JSON)│     │ result (JSON)      │
│ tenant_id        │     │ created_at       │     │ status             │
│ parent_session_id│     └──────────────────┘     │ duration_ms        │
│ token_count      │                               │ created_at         │
│ metadata (JSON)  │                               └────────────────────┘
│ created_at       │
│ updated_at       │     ┌──────────────────┐     ┌──────────────────┐
└──────────────────┘     │  af_artifacts    │     │  af_audit_log    │
                          ├──────────────────┤     ├──────────────────┤
┌──────────────────┐     │ session_id (FK)  │     │ session_id       │
│   af_memory      │     │ name             │     │ event_type       │
├──────────────────┤     │ content (JSON)   │     │ data (JSON)      │
│ session_id       │     │ created_at       │     │ created_at       │
│ key              │     └──────────────────┘     └──────────────────┘
│ value (JSON)     │
│ created_at       │
└──────────────────┘
```

### Backend Selection

```python
from corza_agents import create_repository

# Prototyping — zero dependencies, data in-memory
repo = create_repository("memory")

# Development — persistent file, no server needed
repo = create_repository("sqlite", db_path="agents.db")

# Production — async PostgreSQL with connection pooling
repo = create_repository("postgres", db_url="postgresql+asyncpg://user:pass@host/db")
```

Tables are auto-created on `await repo.initialize()`. Schema version is tracked — warnings are emitted if the database was created by an older version.

---

## Middleware Pipeline

Middleware hooks fire at six points in the ReAct loop:

```
                    ┌──────────────────┐
                    │ before_llm_call  │ ◀── modify messages/tools before LLM
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    LLM Call      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ after_llm_call   │ ◀── inspect/modify LLM response
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ before_tool_call │ ◀── validate, gate, or modify tool args
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Tool Execution  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ after_tool_call  │ ◀── post-process tool results
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ on_turn_complete │ ◀── end-of-turn analytics
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    on_error      │ ◀── error handling and reporting
                    └──────────────────┘
```

### Built-in Middleware

| Middleware | What it does |
|-----------|-------------|
| **ContextCompressionMiddleware** | Ages tool results through 4 tiers (fresh → warm → cold → expired), progressively compressing older results to save context space |
| **RateLimitMiddleware** | Token-bucket rate limiting. Configurable per user, tenant, or session |
| **AuditMiddleware** | Logs every LLM call and tool execution to `af_audit_log` |
| **TokenTrackingMiddleware** | Tracks token usage per session. Estimates cost based on provider pricing |
| **PermissionMiddleware** | Tool-level access control. Uses glob patterns to allow/deny tools per user/role |
| **LoopGuardMiddleware** | Detects infinite loops (same tool called repeatedly with similar args) and forces the agent to stop |

---

## Multi-Agent Architecture

### The Brain Pattern

```
                              Orchestrator
                                  │
                        ┌─────────▼──────────┐
                        │    Brain Agent     │
                        │  (has manage_agent │
                        │   tool)            │
                        └────────┬───────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
      ┌──────────▼──┐  ┌───────▼───────┐  ┌───▼──────────┐
      │  Sub-Agent  │  │  Sub-Agent    │  │  Sub-Agent   │
      │  "researcher"│  │  "analyst"   │  │  "writer"    │
      │  own session │  │  own session │  │  own session │
      │  own tools   │  │  own tools   │  │  own tools   │
      └─────────────┘  └──────────────┘  └──────────────┘
```

**Key design decisions:**

1. **Session isolation** — each sub-agent gets its own session with its own message history. The parent only sees the final result
2. **Tool isolation** — sub-agents only have access to tools declared in their `AgentDefinition`, not the brain's tools
3. **Parallel dispatch** — `spawn_parallel` runs up to `max_parallel_agents` (default: 5) sub-agents concurrently via `asyncio.gather`
4. **Restricted actions** — sub-agents can only call `manage_agent(action="report")`. They cannot spawn other agents
5. **Nuclear stop** — `orchestrator.cancel(session_id)` cascades to all child sessions

### Session Hierarchy

```
Session: brain-001 (status: RUNNING)
├── Session: researcher-001 (parent: brain-001, status: COMPLETED)
├── Session: analyst-001 (parent: brain-001, status: RUNNING)
└── Session: writer-001 (parent: brain-001, status: PENDING)
```

---

## Context Health Monitor

The framework tracks context window usage and takes progressive action:

```
Context usage:  0%──────40%──────80%──85%──90%──100%
                         │        │    │    │
                         │        │    │    └── Hard stop (final turn forced)
                         │        │    └─── Warn agent to wrap up
                         │        └──── LLM summarization of old messages
                         └───── Start compressing tool results
```

Configure thresholds per agent:

```python
from corza_agents import ContextHealthConfig

config = ContextHealthConfig(
    max_tokens=128_000,
    compress_threshold=0.40,
    compact_threshold=0.80,
    warn_threshold=0.85,
    stop_threshold=0.90,
)
```

---

## Error Recovery Strategy

```
                         Error occurs
                              │
                    ┌─────────▼──────────┐
                    │ Is it retryable?   │
                    └──┬──────────────┬──┘
                       │              │
                    Yes│              │No
                       │              │
              ┌────────▼───────┐  ┌──▼────────────────┐
              │ Rate limit?    │  │ Session →          │
              │ Wait + retry   │  │ WAITING_INPUT      │
              │                │  │ Resume on next msg │
              │ Timeout?       │  └───────────────────┘
              │ Backoff + retry│
              │                │
              │ Context full?  │
              │ Compact + retry│
              │                │
              │ Provider down? │
              │ Try fallback   │
              └────────────────┘
```

---

## Streaming Event Flow

```
session.started ──▶ turn.started ──▶ llm.text_delta (×N)
                                  ──▶ llm.tool_call
                                  ──▶ tool.executing
                                  ──▶ tool.result
                                  ──▶ turn.completed
                    ──▶ turn.started ──▶ ...  (loop continues)
                    ──▶ session.completed

For multi-agent:
   ──▶ subagent.started ──▶ [sub-agent events] ──▶ subagent.completed
```

All events are SSE-formatted with `data:` prefix, newline separation, and automatic heartbeat during long operations.

---

## API Layer Separation

The framework cleanly separates business logic from HTTP:

```
AgentService (framework-agnostic)     ◀── Use from anywhere
    │
    ├── FastAPI Router (included)     ◀── Thin HTTP adapter
    ├── CLI adapter                   ◀── You can build this
    └── Any web framework             ◀── You can build this
```

`AgentService` has zero web framework imports. It only depends on the `Orchestrator` and agent definitions:

```python
from corza_agents.api.service import AgentService

service = AgentService(orchestrator, {"brain": brain_def})
session = await service.create_session("brain")
async for event in service.send_message(session.id, "Hello"):
    print(event)
```

---

## Provider Architecture

The `AgentLLM` class normalizes all provider differences behind a single interface:

```
                AgentLLM
                   │
    ┌──────────────┼───────────────┐
    │              │               │
    ▼              ▼               ▼
 OpenAI-      Anthropic       Google
 compatible   Messages API    Gemini API
    │
    ├── openai, groq, cerebras, deepseek,
    │   mistral, xai, fireworks, together,
    │   perplexity, cohere, ollama, lmstudio,
    │   vllm, jan, llamacpp, localai,
    │   lemonade, jellybox, docker
    │
    └── Any custom endpoint via custom_providers
```

Each provider requires its own message format conversion, tool schema translation, and streaming protocol handling. `AgentLLM` handles all of this internally.
