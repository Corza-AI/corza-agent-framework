<p align="center">
  <img src="https://raw.githubusercontent.com/Corza-AI/corza-agent-framework/main/docs/assets/corza-logo.png" alt="Corza AI" width="280" />
</p>

<h2 align="center">Corza Agent Framework</h2>

<p align="center">
  <strong>Drop AI agents into your web app — with streaming, sessions, and a real database.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/corza-agents/"><img src="https://img.shields.io/pypi/v/corza-agents?color=blue" alt="PyPI" /></a>
  <a href="https://pypi.org/project/corza-agents/"><img src="https://img.shields.io/pypi/pyversions/corza-agents" alt="Python" /></a>
  <a href="https://github.com/Corza-AI/corza-agent-framework/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Corza-AI/corza-agent-framework" alt="License" /></a>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> &middot;
  <a href="#-how-it-works">How it works</a> &middot;
  <a href="#-build-a-real-app">Build a real app</a> &middot;
  <a href="#-going-deeper">Going deeper</a> &middot;
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

## Why Corza?

Most agent frameworks are designed for Jupyter notebooks and CLI scripts. The moment you try to put an agent **inside a real product** — with logged-in users, live streaming to a browser, conversations that survive a restart, and a Postgres you already own — you're suddenly writing all the plumbing yourself.

Corza is the plumbing. It's a small, opinionated Python library that gives you:

- A **FastAPI router** with endpoints for sessions, messages, streaming, cancel, and resume — already wired up.
- **SSE streaming** to your frontend that just works (heartbeats, reconnection, disconnect detection).
- **Persistence** to Postgres, SQLite, or memory — tables auto-created, no migrations to babysit.
- **Multi-tenant by default** — every session is scoped to a `user_id` and `tenant_id`.
- **23+ LLM providers** behind one string — `"openai:gpt-4.1"`, `"anthropic:claude-sonnet-4-6"`, `"ollama:qwen3:8b"`.
- **Sub-agents that actually work** — a coordinator can dispatch specialists in parallel and pull their reports back.

You write the tools and the prompts. Corza handles the rest.

---

## 🚀 Quickstart

```bash
pip install "corza-agents[openai]"
```

The smallest useful agent — a tool, an agent definition, and a full HTTP API:

```python
from corza_agents import AgentDefinition, ToolRegistry, create_app, tool

@tool(description="Search the knowledge base")
async def search(query: str) -> str:
    return f"Top result for: {query}"

tools = ToolRegistry()
tools.register_function(search)

app = create_app(
    agents={
        "assistant": AgentDefinition(
            name="assistant",
            model="openai:gpt-4.1",
            tools=["search"],
        )
    },
    tool_registry=tools,
    db_url="postgresql+asyncpg://user:pass@localhost:5432/mydb",
)
# uvicorn app:app --reload
```

That's it. You now have a working agent API:

| Endpoint | What it does |
|----------|--------------|
| `POST /api/agent/sessions` | Start a new conversation |
| `POST /api/agent/sessions/{id}/messages` | Send a message — streams the reply as SSE |
| `GET  /api/agent/sessions/{id}/messages` | Fetch the conversation history |
| `POST /api/agent/sessions/{id}/cancel` | Stop a running session (and any sub-agents) |
| `POST /api/agent/sessions/{id}/resume` | Resume after a failure |
| `GET  /api/agent/health` | Health check |

### Stream the reply to a browser

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
    if (!line.startsWith("data: ")) continue;
    const event = JSON.parse(line.slice(6));
    if (event.data?.text) output.textContent += event.data.text;
  }
}
```

---

## 🧠 How it works

A Corza agent is a small **think → act → think** loop. The LLM decides what tool to call, you run the tool, the result goes back to the LLM, and it decides what to do next — until it produces a final answer.

```mermaid
flowchart LR
    U([User message]) --> L1[LLM]
    L1 -->|wants a tool| T[Tool runs]
    T --> L2[LLM]
    L2 -->|more tools| T
    L2 -->|final answer| R([Reply streams to user])
```

Each lap is a **turn**. Corza persists every message and tool call as it happens, so you can stream live to the browser *and* show the full history when the user comes back tomorrow.

### Where it fits in your app

Corza sits between your FastAPI router and your LLM provider. You can use it as a drop-in router (`create_app`) or wire individual pieces into your existing app.

```mermaid
flowchart TB
    Client([Browser])
    subgraph App["Your FastAPI app"]
        Router["Router<br/><sub>SSE · errors · pagination</sub>"]
        Service["AgentService"]
        Orch["Orchestrator<br/><sub>multi-agent coordination</sub>"]
        Engine["AgentEngine<br/><sub>the think-act loop</sub>"]
        DB[("Postgres / SQLite / memory")]
    end
    Provider([LLM provider])
    Client <-- SSE --> Router --> Service --> Orch --> Engine
    Engine <--> Provider
    Engine --> DB
```

### When one agent isn't enough

For complex work, a **coordinator** agent can spawn specialists, run them in parallel, and pull their reports back together.

```mermaid
flowchart TB
    B([Coordinator])
    R["Researcher"]
    A["Analyst"]
    W["Writer"]
    S([Final answer])
    B --> R --> S
    B --> A --> S
    B --> W --> S
```

```python
result = await orchestrator.run(
    "brain",
    "Investigate Q4 revenue dip and draft a memo",
)
# brain.manage_agent(action="spawn_parallel", tasks=[...])
# → researcher + analyst + writer run concurrently
# → coordinator reads their reports and writes the memo
```

If a sub-agent gets stuck, **nuclear stop** cancels it and all its children at once:

```python
await orchestrator.cancel(session_id)
```

---

## 🛠️ Build a real app

### Define a tool

Type hints become the schema. The LLM sees exactly what arguments it can pass.

```python
from corza_agents import tool, ExecutionContext

@tool(description="Search the database")
async def search(query: str, limit: int = 10) -> dict:
    results = await db.search(query, limit)
    return {"results": results, "count": len(results)}
```

Tools can read the live session (user, tenant, working memory) via `ctx` — it's auto-injected and hidden from the LLM's schema:

```python
@tool(description="Remember something for later this session")
def remember(key: str, value: str, ctx: ExecutionContext) -> str:
    ctx.working_memory.store(key, value)
    return f"Stored '{key}'"
```

### Pick any LLM

Models are just strings — `"provider:model"`. Swap them in one line.

```python
model = "openai:gpt-4.1"
model = "anthropic:claude-sonnet-4-6"
model = "ollama:qwen3:8b"          # free, runs on your laptop
```

If your primary is down, Corza tries fallbacks automatically:

```python
AgentDefinition(
    name="assistant",
    model="anthropic:claude-sonnet-4-6",
    fallback_models=["groq:llama-3.3-70b", "openai:gpt-4.1"],
)
```

<details>
<summary><strong>All 23 supported providers</strong></summary>

| Hosted | Fast inference | Local / self-hosted |
|--------|----------------|---------------------|
| OpenAI · Anthropic · Google · Mistral · Cohere · xAI · DeepSeek · Perplexity | Groq · Cerebras · Fireworks · Together | Ollama · LM Studio · vLLM · llama.cpp · LocalAI · Jan · Lemonade · Jellybox · Docker Model Runner |

Plus any OpenAI-compatible endpoint via `custom_providers={"internal": "https://..."}`.

</details>

### Multi-tenant from day one

Your app handles auth — Corza just receives the IDs and scopes everything to them.

```python
# Set by your auth middleware as HTTP headers:
#   X-User-ID: user_123
#   X-Tenant-ID: acme_corp

# Or programmatically:
session = await service.create_session(
    "assistant",
    user_id="user_123",
    tenant_id="acme_corp",
)
sessions = await service.get_sessions_for_user("user_123", "acme_corp")
```

### Pick your database

Same API across all three — start small, upgrade when ready.

```python
from corza_agents import create_repository

repo = create_repository("memory")                           # dev / tests
repo = create_repository("sqlite", db_path="agents.db")      # single-node
repo = create_repository("postgres", db_url="postgresql+asyncpg://...")  # production
```

Tables (`af_sessions`, `af_messages`, `af_tool_executions`, `af_artifacts`, `af_audit_log`, `af_memory`) are created on startup — no migration step.

---

## 🧩 Going deeper

### The prompt stack

Each agent's "personality" is built from four layers. Keep the system prompt short and identity-level; load knowledge and procedures as separate, swappable pieces.

| Layer | Role | Contents |
|------:|------|----------|
| **1 · System Prompt** | _Principles_ | Who the agent is, how it thinks. Short and permanent. |
| **2 · Knowledge** | _What it knows_ | Project context loaded from `.md` files. |
| **3 · Skills** | _Procedures_ | Step-by-step playbooks, activated per task. |
| **4 · Working Memory** | _Scratch_ | Runtime state built up during the session. |

See [docs/skills.md](docs/skills.md) for the full breakdown.

### Middleware

Plug into the loop at six points (before/after LLM, before/after tool, on turn complete, on error). Useful for logging, gating, or transforming messages.

```python
from corza_agents import BaseMiddleware

class LoggingMiddleware(BaseMiddleware):
    async def before_llm_call(self, messages, tools, context):
        print(f"Turn {context.turn_number}: {len(messages)} messages")
        return messages, tools
```

Built-in middleware you can turn on:

| Middleware | What it does |
|-----------|--------------|
| `ContextCompressionMiddleware` | Ages old tool results through 4 tiers so context stays small |
| `RateLimitMiddleware` | Token-bucket limits per user, tenant, or session |
| `AuditMiddleware` | Logs every LLM call and tool execution |
| `TokenTrackingMiddleware` | Tracks token usage and cost estimates |
| `PermissionMiddleware` | Restrict which tools each user can call |
| `LoopGuardMiddleware` | Breaks out of repeating tool-call loops |

### Error recovery (free)

Failures happen. Corza handles the common ones for you:

| Failure | What Corza does |
|---------|-----------------|
| Rate limit (429) | Waits `retry_after`, then retries |
| Timeout / connection | Exponential backoff up to `max_llm_retries` |
| Context overflow | Auto-compacts the window and retries once |
| Provider down | Falls through `fallback_models` in order |
| Anything else | Session → `WAITING`; resumes on the next message |

### Long conversations

Three layers of defense keep the context window healthy:

1. **Progressive compression** — old tool results decay through `fresh → warm → cold → expired`.
2. **LLM summarization** — at 80% capacity, older messages get summarized.
3. **Health monitoring** — at 85% the agent is warned to wrap up; at 90% it's forced to stop.

```python
from corza_agents import ContextHealthConfig

AgentDefinition(
    name="researcher",
    model="openai:gpt-4.1",
    metadata={"context_health": ContextHealthConfig(
        max_tokens=128_000,
        compress_threshold=0.40,
        compact_threshold=0.80,
    )},
)
```

### Streaming events

Every interesting moment in the loop emits an SSE event. Wire whichever ones you care about into your UI.

| Event | Fires when |
|-------|-----------|
| `session.started` · `session.completed` | A run begins / ends |
| `turn.started` · `turn.completed` | One think-act lap |
| `llm.text_delta` | A chunk of generated text |
| `llm.tool_call` | The LLM asked for a tool |
| `tool.executing` · `tool.result` | A tool ran |
| `subagent.started` · `subagent.completed` | A sub-agent was spawned / finished |
| `error` | Something went wrong |

### Scheduled agents

Run an agent on a cron, once at a specific time, or on an event.

```python
from corza_agents import AgentScheduler, ScheduleEntry

scheduler = AgentScheduler(orchestrator)
scheduler.add(ScheduleEntry(
    name="daily-report",
    agent="analyst",
    message="Generate the daily metrics report",
    cron="0 9 * * *",
    tenant_id="acme_corp",
))
await scheduler.start()
```

### Use Corza inside an existing FastAPI app

You don't have to use `create_app` — inject the service into your own routes:

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

## 📚 Examples

Four runnable examples, smallest to largest:

| File | What it shows | Size |
|------|--------------|-----:|
| [`01_hello_agent.py`](examples/01_hello_agent.py) | One tool, one agent | ~25 lines |
| [`02_custom_tools.py`](examples/02_custom_tools.py) | Sync/async tools, working memory | ~50 lines |
| [`03_multi_agent.py`](examples/03_multi_agent.py) | Coordinator with researcher + writer sub-agents | ~70 lines |
| [`04_web_app.py`](examples/04_web_app.py) | Full FastAPI app with an HTML chat UI | ~100 lines |

```bash
# Free / local
ollama pull qwen3:8b && ollama serve
python examples/01_hello_agent.py

# Or with OpenAI
export OPENAI_API_KEY="sk-..."
python examples/01_hello_agent.py
```

---

## 🔒 Security notes

A few things to know before you ship:

- **Auth is yours.** Corza never authenticates anyone — your app passes `user_id` and `tenant_id` and Corza scopes data to them.
- **Code execution is off by default.** The `CODE` tool type runs Python in a subprocess with no sandbox. Opt in with `CORZA_ALLOW_CODE_EXECUTION=true` only in trusted environments.
- **Runtime registration is off by default.** `POST /tools` and `POST /agents` return 403 unless you set `admin_only=False`.
- **Tool permissions.** Use `PermissionMiddleware` to restrict who can call what:

  ```python
  PermissionMiddleware(rules=[
      PermissionRule(pattern="search_*", allow=True),
      PermissionRule(pattern="admin_*", allow=False),
  ])
  ```

---

## 📁 Project layout

<details>
<summary><strong>Click to expand the full module tree</strong></summary>

```
corza-agent-framework/
├── src/corza_agents/
│   ├── core/              # Engine, LLM client, types, errors
│   ├── api/               # FastAPI router + AgentService
│   ├── orchestrator/      # Multi-agent coordination
│   ├── tools/             # @tool decorator, registry, built-ins
│   ├── skills/            # Skill loading and injection
│   ├── memory/            # Working memory + context management
│   ├── middleware/        # Pipeline hooks (audit, rate-limit, …)
│   ├── persistence/       # Memory / SQLite / Postgres backends
│   ├── streaming/         # SSE event system
│   ├── prompts/           # System prompt templates
│   ├── scheduler/         # Cron / one-shot / event triggers
│   ├── app.py             # create_app() convenience
│   └── dependencies.py    # FastAPI DI helpers
├── examples/              # 01–04 runnable examples
├── tests/                 # 235+ tests across 26 files
└── docs/                  # Extended documentation
```

</details>

---

## 🤝 Contributing

```bash
git clone https://github.com/Corza-AI/corza-agent-framework.git
cd corza-agent-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sqlite]"
pytest
```

Details in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
