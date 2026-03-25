# Architecture

## The ReAct Loop

The core of the framework is a Reason + Act loop with automatic retry:

```
User message
    |
    v
+----------+     +---------+     +--------+
|   LLM    | --> |  Tools  | --> |  LLM   | --> ... --> Final response
+----------+     +---------+     +--------+
    ^                                 |
    |                                 |
    +---- messages persisted ---------+
```

Each iteration is a **turn**. The agent keeps looping until:
- The LLM produces a final text response (no more tool calls)
- `max_turns` is hit
- An error occurs (after `max_llm_retries` retry attempts)

## Components

| Component | What it does |
|-----------|-------------|
| `AgentLLM` | Talks to any LLM provider. Handles streaming, tool schemas, message format conversion. |
| `AgentEngine` | Runs the ReAct loop. Manages sessions, calls LLM with retry, executes tools, emits events. |
| `ToolRegistry` | Stores tools. Generates JSON schemas. Dispatches execution with timeout and retry. |
| `BaseRepository` | Abstract persistence interface. Implemented by Memory, SQLite, and PostgreSQL backends. |
| `Orchestrator` | Wraps `AgentEngine` with sub-agent support. The "brain" pattern. |
| `SkillsManager` | Loads skills from files, functions, URLs, or databases. Injects into system prompt. |
| `WorkingMemory` | Per-session scratch space. Tools can store/retrieve data during a run. |
| `ContextManager` | Keeps the context window under control. Truncates, summarizes, compacts. |
| `BaseMiddleware` | Hook into the loop at 6 points: before/after LLM, before/after tool, on turn complete, on error. |
| `AgentService` | Framework-agnostic service layer. Use with any web framework or CLI. |

## Persistence Backends

```python
from corza_agents import create_repository

repo = create_repository("memory")     # In-memory (no deps, for prototyping)
repo = create_repository("sqlite")     # SQLite file (requires aiosqlite)
repo = create_repository("postgres")   # PostgreSQL (production)
```

All backends store: sessions, messages, tool executions, artifacts, audit log, cross-session memory.

**Database tables** (auto-created by all backends):

| Table | Contents |
|-------|----------|
| `af_sessions` | Agent sessions with status, token counts, metadata |
| `af_messages` | Conversation messages (user, assistant, tool results) |
| `af_tool_executions` | Tool call audit log with inputs, outputs, timing |
| `af_artifacts` | Named outputs stored by agents |
| `af_audit_log` | Middleware audit events |
| `af_memory` | Cross-session long-term memory |

## Error Recovery

The engine retries transient LLM errors automatically:

| Error Type | Behavior |
|-----------|----------|
| Rate limit | Wait `retry_after_seconds`, retry |
| Timeout / connection | Exponential backoff (2^attempt, max 30s), retry up to `max_llm_retries` |
| Context overflow | Compact context window, retry once |
| Non-retryable | Session enters `WAITING_INPUT` — next user message resumes |

Configure via `AgentDefinition(max_llm_retries=3)`.

## Multi-Agent

The `Orchestrator` registers sub-agents. The brain agent gets a `manage_agent` tool that delegates work:

```
Brain Agent
    |
    +-- spawn_parallel([                          ← concurrent dispatch
    |       {"researcher", "find data on X"},
    |       {"analyst", "check the numbers"},
    |   ])
    |       \-- both run in parallel (asyncio.gather)
    |       \-- returns all SubAgentResults at once
    |
    +-- spawn("writer", "summarize these facts")  ← sequential
    |       \-- writer runs in its own session
    |       \-- returns SubAgentResult
    |
    +-- synthesizes results into final answer
```

Each sub-agent gets its own session, its own tool registry, and its own message history. The parent only sees the final result.

**Parallel dispatch**: Up to `max_parallel_agents` (default 5) sub-agents run concurrently via `spawn_parallel`.

**Nuclear stop**: `orchestrator.cancel(session_id)` cancels the parent AND all child sessions. Running loops detect cancellation at the top of each turn and exit immediately.

**Sub-agent restrictions**: Task agents can only call `manage_agent(action="report")`. They cannot spawn, message, or list other agents.

## API Layer

The framework separates business logic from HTTP:

```
AgentService (framework-agnostic)     <-- Use from anywhere
    |
    +-- FastAPI Router (thin adapter)  <-- Optional HTTP layer
    +-- CLI adapter                    <-- You can write this
    +-- Flask adapter                  <-- You can write this
```

```python
from corza_agents.api.service import AgentService

service = AgentService(orchestrator, {"brain": brain_def})
session = await service.create_session("brain")
async for event in service.send_message(session.id, "Hello"):
    print(event)
```

## Streaming Events

Every action in the loop emits a `StreamEvent`:

| Event | When |
|-------|------|
| `session.started` | Run begins |
| `turn.started` | New turn begins |
| `llm.text_delta` | LLM streams a text chunk |
| `llm.tool_call` | LLM requests a tool call |
| `tool.executing` | Tool execution begins |
| `tool.result` | Tool returns a result |
| `subagent.started` | Sub-agent spawned |
| `subagent.completed` | Sub-agent finished |
| `turn.completed` | Turn ends |
| `session.completed` | Run ends |
| `error` | Something went wrong |

Access event data via `event.data` dict. Event type via `event.type.value` string.

## Provider Model Strings

The format is `provider:model_name`. 23+ providers supported out of the box:

```
openai:gpt-5.4                   # OpenAI
anthropic:claude-sonnet-4-6       # Anthropic
google:gemini-3.1-pro             # Google Gemini
groq:llama-3.3-70b-versatile     # Groq (fast inference)
deepseek:deepseek-chat            # DeepSeek
mistral:mistral-large-latest      # Mistral
cerebras:llama-3.3-70b           # Cerebras
fireworks:llama-v3p3-70b-instruct # Fireworks
together:meta-llama/Meta-Llama-3.1-70B  # Together
xai:grok-3                        # xAI Grok
cohere:command-r-plus             # Cohere
perplexity:sonar-pro              # Perplexity
ollama:qwen3:8b                   # Ollama (local)
lmstudio:qwen3-8b                 # LM Studio (local)
vllm:meta-llama/Llama-3.1-8B     # vLLM (self-hosted)
```

Any OpenAI-compatible API works. Register custom endpoints:

```python
llm = AgentLLM(custom_providers={"myhost": "https://my-llm.internal/v1"})
```
