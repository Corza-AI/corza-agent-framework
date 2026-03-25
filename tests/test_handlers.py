"""Tests for tool handlers (code execution, dispatch routing)."""
import os

import pytest

from corza_agents.core.types import (
    ExecutionContext,
    RegisteredTool,
    ToolCall,
    ToolStatus,
    ToolType,
)
from corza_agents.tools.handlers import (
    TOOL_TYPE_HANDLERS,
    dispatch_tool,
    execute_api_tool,
    execute_code_tool,
    execute_db_query_tool,
    execute_workflow_tool,
)

# ══════════════════════════════════════════════════════════════════════
# Code tool — safety gate
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_code_tool_blocked_without_env():
    """Code execution must be rejected when CORZA_ALLOW_CODE_EXECUTION is not set."""
    # Ensure the env var is unset
    os.environ.pop("CORZA_ALLOW_CODE_EXECUTION", None)

    tool_def = RegisteredTool(
        name="run_code",
        description="Execute Python",
        tool_type=ToolType.CODE,
    )
    context = ExecutionContext(session_id="s1")

    result = await execute_code_tool(tool_def, args={"code": "print('hi')"}, context=context)

    assert result["status"] == "error"
    assert "CORZA_ALLOW_CODE_EXECUTION" in result["message"]


@pytest.mark.asyncio
async def test_code_tool_allowed_with_env():
    """Code execution should succeed when the env var is set."""
    tool_def = RegisteredTool(
        name="run_code",
        description="Execute Python",
        tool_type=ToolType.CODE,
    )
    context = ExecutionContext(session_id="s1")

    os.environ["CORZA_ALLOW_CODE_EXECUTION"] = "true"
    try:
        result = await execute_code_tool(
            tool_def,
            args={"code": "import json; print(json.dumps({'result': 42}))"},
            context=context,
        )
        assert result["status"] == "success"
        assert result["result"]["result"] == 42
    finally:
        os.environ.pop("CORZA_ALLOW_CODE_EXECUTION", None)


# ══════════════════════════════════════════════════════════════════════
# Handler dispatch routing
# ══════════════════════════════════════════════════════════════════════

def test_dispatch_tool_routing():
    """TOOL_TYPE_HANDLERS should map all non-function tool types to the correct handlers."""
    assert ToolType.API in TOOL_TYPE_HANDLERS
    assert ToolType.DB_QUERY in TOOL_TYPE_HANDLERS
    assert ToolType.WORKFLOW in TOOL_TYPE_HANDLERS
    assert ToolType.CODE in TOOL_TYPE_HANDLERS

    assert TOOL_TYPE_HANDLERS[ToolType.API] is execute_api_tool
    assert TOOL_TYPE_HANDLERS[ToolType.DB_QUERY] is execute_db_query_tool
    assert TOOL_TYPE_HANDLERS[ToolType.WORKFLOW] is execute_workflow_tool
    assert TOOL_TYPE_HANDLERS[ToolType.CODE] is execute_code_tool


@pytest.mark.asyncio
async def test_dispatch_tool_unknown_type_returns_error():
    """dispatch_tool should return an error ToolResult for unregistered tool types (e.g. FUNCTION)."""
    tool_def = RegisteredTool(
        name="my_func",
        description="A plain function tool",
        tool_type=ToolType.FUNCTION,
    )
    tool_call = ToolCall(tool_name="my_func", arguments={})
    context = ExecutionContext(session_id="s1")

    result = await dispatch_tool(tool_def, tool_call, context)

    assert result.status == ToolStatus.ERROR
    assert "No handler" in result.error
