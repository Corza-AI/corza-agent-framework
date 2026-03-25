"""
01 — Hello Agent

Minimal example: define one tool, create an agent, and stream its response.
This is the simplest possible Corza agent (~25 lines of real code).

No database required — uses in-memory persistence.
"""

import asyncio

from corza_agents import (
    AgentDefinition,
    AgentEngine,
    AgentLLM,
    ToolRegistry,
    create_repository,
    tool,
)


# --- Define a tool --------------------------------------------------------
@tool(description="Greet someone by name")
def greet(name: str) -> str:
    return f"Hello, {name}! Welcome to Corza."


# --- Main ------------------------------------------------------------------
async def main():
    # 1. Set up components — no database needed
    llm = AgentLLM()
    repo = create_repository("memory")  # or "sqlite", "postgres"
    await repo.initialize()

    tools = ToolRegistry()
    tools.register_function(greet)

    # 2. Define the agent — model is required, pick your provider
    agent_def = AgentDefinition(
        name="greeter",
        model="ollama:qwen3:8b",  # or "openai:gpt-5.4", "anthropic:claude-sonnet-4-6"
        system_prompt="You are a friendly greeter. Use the greet tool when asked to say hello.",
        tools=["greet"],
    )

    # 3. Create the engine and run
    engine = AgentEngine(llm=llm, tool_registry=tools, repository=repo)

    async for event in engine.run("session-1", "Say hello to Alice", agent_def):
        if event.data.get("text"):
            print(event.data["text"], end="", flush=True)
        elif event.type.value == "tool.result":
            print(f"\n  [tool] {event.data['tool_name']}: {event.data['status']}")

    print()  # newline at the end


if __name__ == "__main__":
    asyncio.run(main())
