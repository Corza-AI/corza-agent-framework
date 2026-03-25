# Getting Started

## Prerequisites

- Python 3.11+
- An LLM provider (Ollama for local, or API keys for OpenAI/Anthropic/Google)

No database required — the framework runs in-memory by default.

## Setup

```bash
pip install corza-agents

# For local LLM (free, no API keys):
ollama pull qwen3:8b
ollama serve

# For cloud providers:
pip install "corza-agents[openai]"      # or [anthropic], [all]
export OPENAI_API_KEY="sk-..."
```

## Your First Agent

```python
import asyncio
from corza_agents import AgentDefinition, AgentEngine, AgentLLM, ToolRegistry, create_repository, tool

@tool(description="Look up a fact")
async def lookup(topic: str) -> str:
    return f"Here's what I know about {topic}: it's interesting."

async def main():
    llm = AgentLLM()
    repo = create_repository("memory")  # no database needed
    await repo.initialize()

    tools = ToolRegistry()
    tools.register_function(lookup)

    agent = AgentDefinition(
        name="assistant",
        model="ollama:qwen3:8b",  # required — pick your provider
        tools=["lookup"],
    )

    engine = AgentEngine(llm=llm, tool_registry=tools, repository=repo)

    async for event in engine.run("session-1", "Tell me about quantum computing", agent):
        if event.data.get("text"):
            print(event.data["text"], end="")

asyncio.run(main())
```

Run it:

```bash
python my_agent.py
```

The engine will:
1. Send your message to the LLM
2. The LLM decides to call the `lookup` tool
3. The tool returns a result
4. The LLM uses the result to form a response
5. You see it streamed to your terminal

## Persistence Options

Start with in-memory, upgrade when ready:

```python
# Quick prototyping (no deps, data lost on exit)
repo = create_repository("memory")

# Local file persistence
repo = create_repository("sqlite", db_path="agents.db")

# Production PostgreSQL
pip install "corza-agents[postgres]"
repo = create_repository("postgres", db_url="postgresql+asyncpg://localhost:5432/agents")
```

All backends implement the same `BaseRepository` interface — switch at any time.

## Using Cloud Providers

Change the model string and set the API key:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
agent = AgentDefinition(
    name="assistant",
    model="openai:gpt-5.4",  # just change this
    tools=["lookup"],
)
```

Or pass API keys directly:

```python
llm = AgentLLM(api_keys={"openai": "sk-...", "anthropic": "sk-ant-..."})
```

23+ providers supported: `openai`, `anthropic`, `google`, `ollama`, `groq`, `cerebras`, `deepseek`, `mistral`, `xai`, `cohere`, `perplexity`, `fireworks`, `together`, `jan`, `lmstudio`, `llamacpp`, `localai`, `lemonade`, `jellybox`, `docker`, `vllm`, and any OpenAI-compatible endpoint.

## Next

- [Architecture](architecture.md) — how the pieces fit together
- [Skills](skills.md) — the principles + knowledge + skills approach
