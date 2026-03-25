"""Tests for the tool system."""
import pytest

from corza_agents import ExecutionContext, ToolRegistry, tool
from corza_agents.core.types import ToolCall, ToolStatus


@tool(description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Greet someone")
async def greet(name: str) -> str:
    return f"Hello, {name}!"


@tool(description="Uses context")
def with_context(ctx: ExecutionContext, key: str) -> str:
    ctx.working_memory.store(key, "stored")
    return f"stored {key}"


def test_tool_decorator_creates_definition():
    assert hasattr(add, "tool_definition")
    assert add.tool_definition.name == "add"
    assert add.tool_definition.description == "Add two numbers"


def test_tool_json_schema():
    schema = add.tool_definition.to_tool_schema()
    assert schema.name == "add"
    props = schema.parameters.get("properties", {})
    assert "a" in props
    assert "b" in props
    assert props["a"]["type"] == "number"


def test_tool_context_param_excluded_from_schema():
    schema = with_context.tool_definition.to_tool_schema()
    props = schema.parameters.get("properties", {})
    assert "ctx" not in props
    assert "key" in props


def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register_function(add)
    assert reg.has("add")
    assert reg.get("add") is not None


def test_registry_register_many():
    reg = ToolRegistry()
    reg.register_many([add, greet])
    assert reg.has("add")
    assert reg.has("greet")


def test_registry_get_schemas():
    reg = ToolRegistry()
    reg.register_many([add, greet])
    schemas = reg.get_schemas(["add", "greet"])
    assert len(schemas) == 2
    names = {s.name for s in schemas}
    assert names == {"add", "greet"}


@pytest.mark.asyncio
async def test_registry_execute_sync_tool():
    reg = ToolRegistry()
    reg.register_function(add)
    ctx = ExecutionContext(session_id="test")
    result = await reg.execute(
        ToolCall(tool_name="add", arguments={"a": 3, "b": 4}), ctx
    )
    assert result.status == ToolStatus.SUCCESS
    assert result.output == 7.0


@pytest.mark.asyncio
async def test_registry_execute_async_tool():
    reg = ToolRegistry()
    reg.register_function(greet)
    ctx = ExecutionContext(session_id="test")
    result = await reg.execute(
        ToolCall(tool_name="greet", arguments={"name": "Alice"}), ctx
    )
    assert result.status == ToolStatus.SUCCESS
    assert "Alice" in str(result.output)
