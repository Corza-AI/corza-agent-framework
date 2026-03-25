"""
02 — Custom Tools

Demonstrates different ways to create and register tools:
  - Sync and async tool functions
  - Tools with typed parameters
  - Accessing ExecutionContext for working memory
  - Bulk registration with register_many()

No database required — uses in-memory persistence.
"""

import asyncio
from datetime import datetime, timezone

from corza_agents import (
    AgentDefinition,
    AgentEngine,
    AgentLLM,
    ExecutionContext,
    ToolRegistry,
    create_repository,
    tool,
)


# --- Sync tool: simple pure function --------------------------------------
@tool(description="Add two numbers together")
def add(a: float, b: float) -> float:
    return a + b


# --- Async tool: simulates an I/O-bound operation -------------------------
@tool(description="Fetch the current UTC timestamp")
async def get_timestamp() -> str:
    await asyncio.sleep(0.01)  # simulate async I/O
    return datetime.now(timezone.utc).isoformat()


# --- Tool with ExecutionContext: read/write working memory -----------------
@tool(description="Store a key-value pair in working memory for later use")
def remember(ctx: ExecutionContext, key: str, value: str) -> str:
    """The `ctx` parameter is injected automatically by the engine."""
    ctx.working_memory.store(key, value)
    return f"Stored '{key}' in working memory."


@tool(description="Recall a value previously stored in working memory")
def recall(ctx: ExecutionContext, key: str) -> str:
    value = ctx.working_memory.get(key)
    if value is None:
        return f"No value found for key '{key}'."
    return f"Recalled '{key}': {value}"


# --- Main ------------------------------------------------------------------
async def main():
    llm = AgentLLM()
    repo = create_repository("memory")
    await repo.initialize()

    # Register tools in bulk with register_many()
    tools = ToolRegistry()
    tools.register_many([add, get_timestamp, remember, recall])

    agent_def = AgentDefinition(
        name="tool_demo",
        model="ollama:qwen3:8b",  # required — pick your provider
        system_prompt=(
            "You are a helpful assistant with math, time, and memory tools. "
            "Demonstrate each tool when asked."
        ),
        tools=["add", "get_timestamp", "remember", "recall"],
    )

    engine = AgentEngine(llm=llm, tool_registry=tools, repository=repo)

    async for event in engine.run(
        "session-1",
        "Add 42 and 58, then remember the result as 'answer'. "
        "After that, recall the answer and tell me the current time.",
        agent_def,
    ):
        if event.data.get("text"):
            print(event.data["text"], end="", flush=True)
        elif event.type.value == "tool.result":
            print(f"\n  [tool] {event.data['tool_name']}: {event.data['status']}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
