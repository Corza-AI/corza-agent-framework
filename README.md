# Corza Agent Framework

**The agent framework built for web applications.**

Every agent framework out there is built for scripts and notebooks. When you need an agent inside a real web app — with users, sessions, SSE streaming, and a database — you're on your own.

Corza fixes that. It lives inside your FastAPI app, shares your PostgreSQL, streams to your frontend over SSE, and knows about your users and tenants.

```
pip install "corza-agents[openai]"
```

## 30 Seconds to an Agent API

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
        model="openai:gpt-4.1",    # you choose the model
        tools=["search"],
    )},
    tool_registry=tools,
    db_url="postgresql+asyncpg://user:pass@localhost:5432/mydb",
)
# uvicorn app:app
```

That's it. You now have:

```
POST   /api/agent/sessions                     → create session
POST   /api/agent/sessions/{id}/messages        → send message (SSE stream)
GET    /api/agent/sessions/{id}/messages        → message history
POST   /api/agent/sessions/{id}/cancel          → nuclear stop (cancels session + all sub-agents)
POST   /api/agent/sessions/{id}/resume          → resume failed session
DELETE /api/agent/sessions/{id}                 → delete session
GET    /api/agent/health                        → health check
```

## Stream to Your Frontend

```javascript
const res = await fetch(`/api/agent/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({content: "What is AI?", stream: true}),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value).split("\n")) {
        if (line.startsWith("data: ")) {
            const event = JSON.parse(line.slice(6));
            if (event.data?.text) document.getElementById("output").textContent += event.data.text;
        }
    }
}
```

## Architecture

```
Your FastAPI App
├── Router (SSE streaming, structured errors, pagination)
│   └── AgentService (stateless, no web imports)
│       └── Orchestrator (sub-agent delegation)
│           └── AgentEngine (ReAct loop)
│               ├── LLM (23 providers + fallback chain)
│               ├── Tools (@tool decorator, auto JSON schema)
│               └── Memory (working memory + context compression)
│
├── Middleware Pipeline
│   ├── ContextCompression — 4-tier tool result aging
│   ├── RateLimit — per-user/tenant token bucket
│   ├── Audit — logs every LLM + tool call
│   ├── TokenTracking — cost estimation
│   ├── Permission — tool-level access control
│   └── Custom — subclass BaseMiddleware
│
├── Context Health Monitor
│   ├── 0.40 → start compression
│   ├── 0.80 → LLM summarization
│   ├── 0.85 → warn agent to wrap up
│   └── 0.90 → hard stop (final turn)
│
└── Persistence (PostgreSQL / SQLite / InMemory)
    └── 7 tables: sessions, messages, tool_executions, artifacts, audit, memory, schema_version
```

## Install

```bash
pip install corza-agents                       # core (FastAPI + PostgreSQL)
pip install "corza-agents[openai]"             # + OpenAI
pip install "corza-agents[anthropic]"          # + Anthropic
pip install "corza-agents[all]"                # everything
```

## 23 LLM Providers

No default model — you choose. Format: `provider:model_name`.

```python
model="openai:gpt-4.1"               # OpenAI
model="anthropic:claude-sonnet-4-6"   # Anthropic
model="google:gemini-2.5-pro"         # Google
model="groq:llama-3.3-70b-versatile"  # Groq (fast)
model="ollama:qwen3:8b"              # Ollama (local, free)
model="deepseek:deepseek-chat"        # DeepSeek
model="cerebras:llama-3.3-70b"        # Cerebras (fast)
```

Also: Mistral, xAI, Cohere, Perplexity, Fireworks, Together, Jan, LM Studio, llama.cpp, LocalAI, Lemonade, Jellybox, Docker Model Runner, vLLM.

**Fallback chain** — if your primary provider is down:

```python
agent = AgentDefinition(
    name="assistant",
    model="anthropic:claude-sonnet-4-6",
    fallback_models=["groq:llama-3.3-70b", "cerebras:llama-3.3-70b"],
)
```

**Custom providers** — any OpenAI-compatible endpoint:

```python
llm = AgentLLM(custom_providers={"internal": "https://llm.internal.company/v1"})
# model="internal:my-fine-tuned-model"
```

## Tools

```python
from corza_agents import tool, ExecutionContext

@tool(description="Search the database")
async def search(query: str, limit: int = 10) -> dict:
    results = await db.search(query, limit)
    return {"results": results, "count": len(results)}

# Tools can access session context
@tool(description="Store a finding")
def remember(key: str, value: str, ctx: ExecutionContext) -> str:
    ctx.working_memory.store(key, value)
    return f"Stored '{key}'"
```

`ctx` is auto-injected. It doesn't appear in the tool's JSON schema.

## Users & Tenants

Sessions are scoped to users and tenants. Your app handles auth:

```python
# Option 1: Headers (set by your auth middleware)
# X-User-ID: user_123
# X-Tenant-ID: acme_corp

# Option 2: Programmatic
session = await service.create_session("assistant", user_id="user_123", tenant_id="acme_corp")
sessions = await service.get_sessions_for_user("user_123", "acme_corp")
```

## Middleware

Hook into the agent loop at 6 points:

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

**Built-in middleware:**

| Middleware | What it does |
|-----------|-------------|
| `ContextCompressionMiddleware` | 4-tier progressive compression of old tool results |
| `RateLimitMiddleware` | Token-bucket rate limiting per user/tenant/session |
| `AuditMiddleware` | Logs every LLM call and tool execution to DB |
| `TokenTrackingMiddleware` | Tracks token usage and cost per session |
| `PermissionMiddleware` | Tool-level access control with pattern matching |

## Multi-Agent

```python
from corza_agents import Orchestrator

orchestrator = Orchestrator(llm=llm, tool_registry=tools, repository=repo)
orchestrator.register_sub_agent("researcher", AgentDefinition(
    name="researcher", model="openai:gpt-4.1",
    tools=["search_web", "extract_facts"],
))
orchestrator.register_sub_agent("writer", AgentDefinition(
    name="writer", model="openai:gpt-4.1",
    tools=["write_report"],
))
await orchestrator.initialize()

# The brain agent delegates to sub-agents automatically
async for event in orchestrator.run("s1", "Research AI trends", brain_agent):
    print(event.to_sse())
```

**Parallel dispatch** — the orchestrator can run up to 5 sub-agents concurrently (configurable via `max_parallel_agents`):

```python
brain = AgentDefinition(
    name="brain", model="openai:gpt-4.1",
    max_parallel_agents=5,  # default: 5, max: 10
)
```

The orchestrator's prompt instructs it to use `spawn_parallel` for concurrent investigation threads.

**Nuclear stop** — cancel a session and all its sub-agents:

```python
count = await orchestrator.cancel(session_id)  # cancels parent + all children
# Or via HTTP:
# POST /api/agent/sessions/{id}/cancel → {"sessions_cancelled": 3}
```

## Persistence

```python
from corza_agents import create_repository

repo = create_repository("postgres", db_url="postgresql+asyncpg://...")  # production
repo = create_repository("sqlite", db_path="agents.db")                  # dev
repo = create_repository("memory")                                        # tests
```

Tables are auto-created on startup. Schema version is tracked — you'll get a warning if the DB schema is outdated.

## Error Recovery

Built-in, no configuration needed:

- **Rate limits** → waits `retry_after_seconds`, retries
- **Timeouts** → configurable `llm_timeout_seconds` (default 120s), exponential backoff
- **Context overflow** → auto-compacts conversation, retries
- **Failed turns** → session enters `WAITING_INPUT` — resume with `POST /resume`
- **Provider down** → tries `fallback_models` in order

## Context Management

Long conversations are handled automatically:

1. **Progressive compression** — old tool results are compressed by age (fresh → warm → cold → expired)
2. **LLM summarization** — when context hits 80% capacity, old messages are summarized
3. **Health monitoring** — at 80% the agent is warned to wrap up; at 90% it's forced to stop

Configure per agent:

```python
from corza_agents import ContextHealthConfig

agent = AgentDefinition(
    name="researcher",
    model="openai:gpt-4.1",
    metadata={"context_health": ContextHealthConfig(max_tokens=128_000)},
)
```

## FastAPI Dependency Injection

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

## Examples

| Example | What it shows |
|---------|--------------|
| [`01_hello_agent.py`](examples/01_hello_agent.py) | Minimal agent, one tool, ~25 lines |
| [`02_custom_tools.py`](examples/02_custom_tools.py) | Sync/async tools, working memory |
| [`03_multi_agent.py`](examples/03_multi_agent.py) | Orchestrator with sub-agents |
| [`04_web_app.py`](examples/04_web_app.py) | **Complete web app with HTML chat UI** |

## Security

### Authentication

The framework does **not** implement authentication. Your FastAPI app handles auth — the framework accepts `user_id` and `tenant_id` as pass-through context that scopes sessions and data:

```python
# In your FastAPI app — you handle auth, the framework receives the result
@app.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    events = service.send_message(session_id, request.content, metadata={
        "user_id": user.id,
        "tenant_id": user.tenant_id,
    })
    return await sse_response(request, events)
```

### Code Execution

The `CODE` tool type runs Python in a subprocess with **no sandboxing** beyond a timeout. It is **disabled by default**. To enable:

```bash
export CORZA_ALLOW_CODE_EXECUTION=true  # Required. Only enable in trusted environments.
```

### Runtime Registration

`POST /tools` and `POST /agents` endpoints are **disabled by default** (return 403). To enable runtime registration:

```python
router = create_agent_router(orchestrator, agents, admin_only=False)
```

## Troubleshooting

### Context Overflow

If agents hit "context too large" errors, the framework has 3-layer defense built in:
1. **Auto-truncation** — old tool arguments are shortened automatically
2. **Auto-summarization** — old messages are summarized by the LLM when context exceeds 80%
3. **Manual compaction** — agents can call `manage_context()` to trigger compaction

To tune thresholds, pass `ContextHealthConfig` in agent metadata:

```python
from corza_agents.memory.health import ContextHealthConfig

agent = AgentDefinition(
    ...,
    metadata={"context_health": ContextHealthConfig(
        compress_threshold=0.40,  # Start compressing at 40% usage
        compact_threshold=0.80,   # Auto-summarize at 80%
    )}
)
```

### Provider Errors

The framework automatically retries transient LLM errors with exponential backoff (up to `max_llm_retries`, default 3). If a primary model fails, it tries each model in `fallback_models` in order:

```python
agent = AgentDefinition(
    model="openai:gpt-4.1",
    fallback_models=["groq:llama-3.3-70b", "cerebras:llama-3.3-70b"],
)
```

### Migration Warnings

If you see "schema version mismatch" on startup, the database tables were created by an older version. Back up your database and re-run `initialize()` — the framework creates missing tables but does not alter existing ones.

## License

MIT — see [LICENSE](LICENSE).
