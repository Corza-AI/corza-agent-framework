<p align="center">
  <img src="https://raw.githubusercontent.com/Corza-AI/corza-agent-framework/main/docs/assets/corza-logo.png" alt="Corza AI" width="280" />
</p>

<h2 align="center">Agent Framework</h2>

<p align="center">
  <strong>The agent framework built for web applications.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/corza-agents/"><img src="https://img.shields.io/pypi/v/corza-agents?color=blue" alt="PyPI" /></a>
  <a href="https://pypi.org/project/corza-agents/"><img src="https://img.shields.io/pypi/pyversions/corza-agents" alt="Python" /></a>
  <a href="https://github.com/Corza-AI/corza-agent-framework/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Corza-AI/corza-agent-framework" alt="License" /></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="docs/architecture.md">Architecture</a> &middot;
  <a href="docs/getting-started.md">Getting Started</a> &middot;
  <a href="docs/skills.md">Skills &amp; Knowledge</a> &middot;
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

Every agent framework out there is built for scripts and notebooks. When you need an agent inside a real web app — with users, sessions, SSE streaming, and a database — you're on your own.

Corza fixes that. It lives inside your FastAPI app, shares your PostgreSQL, streams to your frontend over SSE, and knows about your users and tenants.

```
pip install "corza-agents[openai]"
```

---

## Why Corza?

| Challenge | How Corza Solves It |
|-----------|-------------------|
| **No streaming** — most frameworks return full responses | Real-time SSE streaming with heartbeat, reconnection, and client disconnect detection |
| **No persistence** — conversations vanish on restart | PostgreSQL, SQLite, or in-memory backends with auto-created tables |
| **No multi-tenancy** — agents don't know about users | Built-in user/tenant scoping on every session and query |
| **Single model lock-in** — tied to one LLM provider | 23+ providers with a `provider:model` string. Swap in one line |
| **No error recovery** — one failure kills the session | Auto-retry with backoff, fallback chains, context overflow recovery |
| **No sub-agents** — orchestration is DIY | Built-in orchestrator with parallel dispatch and nuclear stop |

---

## Quickstart

### 30 Seconds to an Agent API

```python
from corza_agents import AgentDefinition, ToolRegistry, create_app, tool

@tool(description="Search the knowledge base")
async def search(query: str) -> str:
    return f"Results for: {query}"

tools = ToolRegistry()
tools.register_function(search)

app = create_app(
    agents={"assistant": AgentDefinition(
        name="assistant",
        model="openai:gpt-4.1",
        tools=["search"],
    )},
    tool_registry=tools,
    db_url="postgresql+asyncpg://user:pass@localhost:5432/mydb",
)
# uvicorn app:app
```

That gives you a full REST API:

```
POST   /api/agent/sessions                     → create session
POST   /api/agent/sessions/{id}/messages        → send message (SSE stream)
GET    /api/agent/sessions/{id}/messages        → message history
POST   /api/agent/sessions/{id}/cancel          → cancel session + all sub-agents
POST   /api/agent/sessions/{id}/resume          → resume failed session
DELETE /api/agent/sessions/{id}                 → delete session
GET    /api/agent/health                        → health check
```

### Stream to Your Frontend

```javascript
const res = await fetch(`/api/agent/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: "What is AI?", stream: true }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value).split("\n")) {
        if (line.startsWith("data: ")) {
            const event = JSON.parse(line.slice(6));
            if (event.data?.text) {
                document.getElementById("output").textContent += event.data.text;
            }
        }
    }
}
```

---

## How It Works

### High-Level Architecture

```mermaid
flowchart TB
    Client([Browser / Client])
    subgraph App["Your FastAPI App"]
        Router["Router<br/><sub>SSE streaming · structured errors</sub>"]
        Service["AgentService<br/><sub>stateless, framework-agnostic</sub>"]
        Orch["Orchestrator<br/><sub>brain pattern · sub-agent mgmt</sub>"]
        Engine["AgentEngine<br/><sub>ReAct loop</sub>"]
        LLM[LLM]
        Tools[Tools]
        Mem[Memory]
        MW["Middleware Pipeline<br/><sub>compression · rate limit · audit · tokens · permissions · loop guard</sub>"]
        DB[("Persistence<br/><sub>Postgres · SQLite · in-memory</sub>")]
    end
    Client -- HTTP / SSE --> Router --> Service --> Orch --> Engine
    Engine --> LLM
    Engine --> Tools
    Engine --> Mem
    Engine -. hooks .-> MW
    Engine --> DB
```

### The ReAct Loop

Every agent runs a **Reason + Act** loop. Each iteration is a **turn**, and the agent keeps looping until the LLM produces a final text response (no more tool calls), `max_turns` is reached, or an unrecoverable error occurs.

```mermaid
flowchart LR
    U([User message]) --> L1[LLM]
    L1 -->|tool calls| T[Tools]
    T --> L2[LLM]
    L2 -->|more tools| T
    L2 -->|final text| R([Response])
```

### Multi-Agent Orchestration

The Orchestrator implements a **brain pattern** — one coordinator agent delegates to specialized sub-agents.

```mermaid
flowchart TB
    B(["Brain Agent"])
    R["Researcher<br/><sub>search · extract</sub>"]
    A["Analyst<br/><sub>query · compute</sub>"]
    W["Writer<br/><sub>format · compose</sub>"]
    S(["Synthesized Answer"])
    B --> R --> S
    B --> A --> S
    B --> W --> S
```

**Parallel dispatch** — up to 5 sub-agents run concurrently (configurable):

```python
brain = AgentDefinition(
    name="brain",
    model="openai:gpt-4.1",
    max_parallel_agents=5,  # default: 5, max: 10
)
```

**Nuclear stop** — cancel a session and all its children:

```python
count = await orchestrator.cancel(session_id)
# POST /api/agent/sessions/{id}/cancel → {"sessions_cancelled": 3}
```

### The Prompt Stack

Corza separates agent identity into four composable layers:

| Layer | Role | Contents |
|------:|------|----------|
| **1 · System Prompt** | _Principles_ | Who the agent is, how it thinks. Short, permanent, identity-level. |
| **2 · Knowledge** | _What it knows_ | Project context loaded from `.md` files. |
| **3 · Skills** | _Procedures_ | Step-by-step playbooks, activated per task. |
| **4 · Working Memory** | _Scratch_ | Runtime state built up during the session. |

See [docs/skills.md](docs/skills.md) for the full explanation.

---

## Installation

```bash
# Core framework (FastAPI + async PostgreSQL)
pip install corza-agents

# With your preferred LLM provider
pip install "corza-agents[openai]"        # OpenAI
pip install "corza-agents[anthropic]"     # Anthropic
pip install "corza-agents[google]"        # Google Gemini
pip install "corza-agents[all]"           # Everything
```

**Requirements:** Python 3.11+

---

## 23 LLM Providers

No default model. You choose at runtime with a `provider:model` string:

```python
model = "openai:gpt-4.1"
model = "anthropic:claude-sonnet-4-6"
model = "google:gemini-2.5-pro"
model = "groq:llama-3.3-70b-versatile"
model = "ollama:qwen3:8b"  # local, free
```

<details>
<summary><strong>All supported providers</strong></summary>

| Hosted | Fast inference | Local / self-hosted |
|--------|----------------|---------------------|
| OpenAI · Anthropic · Google · Mistral · Cohere · xAI · DeepSeek · Perplexity | Groq · Cerebras · Fireworks · Together | Ollama · LM Studio · vLLM · llama.cpp · LocalAI · Jan · Lemonade · Jellybox · Docker Model Runner |

Plus any OpenAI-compatible endpoint via `custom_providers`.

</details>

### Fallback Chains

If your primary provider goes down, the framework tries alternatives in order:

```python
agent = AgentDefinition(
    name="assistant",
    model="anthropic:claude-sonnet-4-6",
    fallback_models=["groq:llama-3.3-70b", "cerebras:llama-3.3-70b"],
)
```

### Custom Providers

Any OpenAI-compatible API:

```python
llm = AgentLLM(custom_providers={"internal": "https://llm.internal.company/v1"})
# Then use: model="internal:my-fine-tuned-model"
```

---

## Tools

Define tools with the `@tool` decorator. The framework auto-generates JSON schemas from type annotations:

```python
from corza_agents import tool, ExecutionContext

@tool(description="Search the database")
async def search(query: str, limit: int = 10) -> dict:
    results = await db.search(query, limit)
    return {"results": results, "count": len(results)}

# Tools can access session context via ExecutionContext
@tool(description="Store a finding in working memory")
def remember(key: str, value: str, ctx: ExecutionContext) -> str:
    ctx.working_memory.store(key, value)
    return f"Stored '{key}'"
```

The `ctx` parameter is auto-injected by the engine — it does not appear in the tool's JSON schema sent to the LLM.

### Bulk Registration

```python
tools = ToolRegistry()
tools.register_many([search, remember, calculate, fetch_data])
```

---

## Users & Tenants

Every session is scoped to a user and tenant. Your app handles authentication — the framework accepts IDs as pass-through context:

```python
# Option 1: HTTP headers (set by your auth middleware)
# X-User-ID: user_123
# X-Tenant-ID: acme_corp

# Option 2: Programmatic
session = await service.create_session(
    "assistant",
    user_id="user_123",
    tenant_id="acme_corp",
)
sessions = await service.get_sessions_for_user("user_123", "acme_corp")
```

---

## Middleware

Hook into the agent loop at six points — before/after LLM calls, before/after tool execution, on turn complete, and on error:

```python
from corza_agents import BaseMiddleware

class LoggingMiddleware(BaseMiddleware):
    async def before_llm_call(self, messages, tools, context):
        print(f"Turn {context.turn_number}: calling LLM with {len(messages)} messages")
        return messages, tools

    async def after_tool_call(self, tool_call, result, context):
        print(f"Tool {tool_call.tool_name}: {result.status.value}")
        return result
```

### Built-in Middleware

| Middleware | Purpose |
|-----------|---------|
| **ContextCompressionMiddleware** | 4-tier progressive compression of old tool results (fresh → warm → cold → expired) |
| **RateLimitMiddleware** | Token-bucket rate limiting per user, tenant, or session |
| **AuditMiddleware** | Logs every LLM call and tool execution to the database |
| **TokenTrackingMiddleware** | Tracks token usage and estimates cost per session |
| **PermissionMiddleware** | Tool-level access control with glob pattern matching |
| **LoopGuardMiddleware** | Detects and breaks infinite tool-calling loops |

---

## Persistence

Three backends, one interface. Start with in-memory, upgrade when ready:

```python
from corza_agents import create_repository

repo = create_repository("memory")     # No deps, data lost on exit
repo = create_repository("sqlite", db_path="agents.db")  # Local file
repo = create_repository("postgres", db_url="postgresql+asyncpg://...")  # Production
```

All backends store the same data:

| Table | Contents |
|-------|----------|
| `af_sessions` | Sessions with status, token counts, user/tenant metadata |
| `af_messages` | Conversation messages (user, assistant, tool results) |
| `af_tool_executions` | Tool call audit log with inputs, outputs, timing |
| `af_artifacts` | Named outputs stored by agents |
| `af_audit_log` | Middleware audit events |
| `af_memory` | Cross-session long-term memory |

Tables are auto-created on startup. Schema versions are tracked automatically.

---

## Error Recovery

Built-in, no configuration needed:

| Failure | Recovery |
|---------|----------|
| **Rate limit (429)** | Wait `retry_after`, then retry |
| **Timeout / connection** | Exponential backoff up to `max_llm_retries` |
| **Context overflow** | Auto-compact window, retry once |
| **Provider down** | Try `fallback_models` in order |
| **Non-retryable error** | Session → `WAITING`, resume on next message |

---

## Context Management

Long conversations are handled automatically with a three-layer defense:

1. **Progressive compression** — old tool results age through 4 tiers (fresh → warm → cold → expired), each more aggressively compressed
2. **LLM summarization** — when context hits 80% capacity, older messages are summarized by the LLM
3. **Health monitoring** — at 85% the agent is warned to wrap up; at 90% it's forced to stop

```python
from corza_agents import ContextHealthConfig

agent = AgentDefinition(
    name="researcher",
    model="openai:gpt-4.1",
    metadata={"context_health": ContextHealthConfig(
        max_tokens=128_000,
        compress_threshold=0.40,   # Start compressing tool results
        compact_threshold=0.80,    # Trigger LLM summarization
    )},
)
```

---

## Streaming Events

Every action in the ReAct loop emits a `StreamEvent` over SSE:

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

---

## FastAPI Dependency Injection

Use Corza as a service inside your existing FastAPI routes:

```python
from corza_agents.dependencies import get_service, get_user_context

@router.post("/analyze")
async def analyze(
    service: AgentService = Depends(get_service),
    user: UserContext = Depends(get_user_context),
):
    session = await service.create_session("analyst", user.user_id, user.tenant_id)
    async for event in service.send_message(session.id, "Analyze Q4 revenue"):
        yield event.to_sse()
```

---

## Scheduler

Run agents on a schedule — cron, one-time, or event-triggered:

```python
from corza_agents import AgentScheduler, ScheduleEntry

scheduler = AgentScheduler(orchestrator)
scheduler.add(ScheduleEntry(
    name="daily-report",
    agent="analyst",
    message="Generate the daily metrics report",
    cron="0 9 * * *",  # Every day at 9 AM
    tenant_id="acme_corp",
))
await scheduler.start()
```

---

## Examples

| Example | Description | Lines |
|---------|-------------|-------|
| [`01_hello_agent.py`](examples/01_hello_agent.py) | Minimal agent with one tool | ~25 |
| [`02_custom_tools.py`](examples/02_custom_tools.py) | Sync/async tools, working memory, bulk registration | ~50 |
| [`03_multi_agent.py`](examples/03_multi_agent.py) | Orchestrator with researcher + writer sub-agents | ~70 |
| [`04_web_app.py`](examples/04_web_app.py) | Complete FastAPI app with HTML chat UI | ~100 |

Run any example:

```bash
# With Ollama (free, local)
ollama pull qwen3:8b && ollama serve
python examples/01_hello_agent.py

# With OpenAI
export OPENAI_API_KEY="sk-..."
python examples/01_hello_agent.py  # change model to "openai:gpt-4.1"
```

---

## Project Structure

<details>
<summary><strong>Click to expand the full module tree</strong></summary>

```
corza-agent-framework/
├── src/corza_agents/
│   ├── core/              # Engine, LLM client, types, error hierarchy
│   │   ├── engine.py      # AgentEngine — the ReAct loop
│   │   ├── llm.py         # AgentLLM — 23+ provider adapters
│   │   ├── types.py       # All dataclasses and enums
│   │   └── errors.py      # Typed error hierarchy
│   ├── api/               # HTTP layer
│   │   ├── router.py      # FastAPI router (thin adapter)
│   │   ├── service.py     # AgentService (framework-agnostic)
│   │   └── schemas.py     # Request/response Pydantic models
│   ├── orchestrator/      # Multi-agent coordination
│   │   ├── orchestrator.py # Brain pattern with sub-agent dispatch
│   │   └── sub_agent.py   # SubAgentRunner isolation
│   ├── tools/             # Tool system
│   │   ├── decorators.py  # @tool decorator
│   │   ├── registry.py    # ToolRegistry with JSON schema gen
│   │   ├── builtin.py     # Built-in tools (manage_agent, etc.)
│   │   └── handlers.py    # Tool type dispatch (function, code, API)
│   ├── skills/            # Skill loading and injection
│   │   └── manager.py     # SkillsManager (file, URL, DB, function)
│   ├── memory/            # Context management
│   │   ├── working.py     # Per-session working memory
│   │   ├── context.py     # ContextManager (truncation, summarization)
│   │   └── health.py      # Context health monitoring
│   ├── middleware/         # Pipeline hooks
│   │   ├── base.py        # BaseMiddleware interface
│   │   ├── audit.py       # AuditMiddleware
│   │   ├── rate_limit.py  # RateLimitMiddleware
│   │   ├── token_tracking.py
│   │   ├── permissions.py # PermissionMiddleware
│   │   ├── context_compression.py
│   │   └── loop_guard.py  # LoopGuardMiddleware
│   ├── persistence/       # Storage backends
│   │   ├── base.py        # BaseRepository ABC
│   │   ├── memory.py      # InMemoryRepository
│   │   ├── sqlite.py      # SQLiteRepository
│   │   ├── repository.py  # PostgresRepository
│   │   ├── models.py      # SQLAlchemy table definitions
│   │   └── factory.py     # create_repository()
│   ├── streaming/         # SSE event system
│   │   ├── events.py      # StreamEvent definitions
│   │   └── sse.py         # SSE response helpers
│   ├── prompts/           # System prompt construction
│   │   └── templates.py   # Prompt templates and builders
│   ├── scheduler/         # Scheduled agent execution
│   │   ├── scheduler.py   # AgentScheduler
│   │   └── models.py      # ScheduleEntry
│   ├── app.py             # create_app() convenience
│   └── dependencies.py    # FastAPI DI helpers
├── examples/              # Runnable examples (01–04)
├── tests/                 # 235+ tests across 26 files
├── docs/                  # Extended documentation
└── pyproject.toml         # Package metadata
```

</details>

---

## Security

### Authentication

The framework does **not** implement authentication. Your app handles auth — the framework receives `user_id` and `tenant_id` as pass-through context that scopes sessions and data.

### Code Execution

The `CODE` tool type runs Python in a subprocess with **no sandboxing** beyond a timeout. It is **disabled by default**:

```bash
export CORZA_ALLOW_CODE_EXECUTION=true  # Only enable in trusted environments
```

### Runtime Registration

`POST /tools` and `POST /agents` endpoints return **403 by default**. Opt-in:

```python
router = create_agent_router(orchestrator, agents, admin_only=False)
```

### Tool Permissions

Use `PermissionMiddleware` to restrict which tools users can access:

```python
from corza_agents import PermissionMiddleware, PermissionRule

middleware = PermissionMiddleware(rules=[
    PermissionRule(pattern="search_*", allow=True),
    PermissionRule(pattern="admin_*", allow=False),
])
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code style guidelines.

```bash
git clone https://github.com/Corza-AI/corza-agent-framework.git
cd corza-agent-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sqlite]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
