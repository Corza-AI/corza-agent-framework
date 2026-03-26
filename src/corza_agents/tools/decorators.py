"""
Corza Agent Framework — Tool Decorators

@tool decorator for defining tools as simple Python functions.
Inspired by CrewAI and Pydantic AI's decorator patterns.
"""
import asyncio
import inspect
import types
from collections.abc import Callable
from functools import wraps
from typing import Any, Union, get_args, get_origin, get_type_hints

from corza_agents.core.types import (
    ExecutionContext,
    RegisteredTool,
    ToolType,
)

# Python type → JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _unwrap_optional(tp: Any) -> Any:
    """Unwrap Optional[X] / Union[X, None] / X | None to get the inner type."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _python_type_to_json(tp: Any) -> str:
    tp = _unwrap_optional(tp)
    origin = get_origin(tp) or getattr(tp, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return _TYPE_MAP.get(tp, "string")


def _build_json_schema(fn: Callable) -> dict[str, Any]:
    """
    Introspect function signature to build JSON Schema for tool parameters.
    Skips 'ctx'/'context' parameters (injected by framework).
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("ctx", "context", "self", "cls"):
            continue
        if hints.get(param_name) is ExecutionContext:
            continue

        tp = hints.get(param_name, str)
        json_type = _python_type_to_json(tp)

        prop: dict[str, Any] = {"type": json_type}

        if param.default is not inspect.Parameter.empty and param.default is not None:
            prop["default"] = param.default
        else:
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Use the parameter name as description by default
        prop["description"] = param_name.replace("_", " ")

        properties[param_name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    name: str | None = None,
    description: str | None = None,
    tool_type: ToolType = ToolType.FUNCTION,
    permission_level: str = "auto_approve",
    timeout_seconds: int = 30,
    retry_max: int = 0,
    tags: list[str] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register a function as an agent tool.

    Usage:
        @tool(name="search_db", description="Search the case database")
        async def search_db(query: str, limit: int = 10, ctx: ExecutionContext) -> dict:
            results = await do_search(query, limit)
            return {"results": results}

    The decorated function gains a `.tool_definition` attribute containing
    the RegisteredTool instance, which can be added to a ToolRegistry.

    Parameters named 'ctx' or 'context' with type ExecutionContext are
    automatically injected by the framework — they don't appear in the
    tool's JSON schema.
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0] or tool_name

        # Use explicit schema if provided, otherwise auto-generate from signature
        resolved_schema = json_schema if json_schema is not None else _build_json_schema(fn)

        is_async = asyncio.iscoroutinefunction(fn)

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            if is_async:
                return await fn(*args, **kwargs)
            return await asyncio.to_thread(fn, *args, **kwargs)

        registered = RegisteredTool(
            name=tool_name,
            description=tool_desc,
            tool_type=tool_type,
            json_schema=resolved_schema,
            permission_level=permission_level,
            timeout_seconds=timeout_seconds,
            retry_max=retry_max,
            tags=tags or [],
            handler=async_wrapper,
        )

        async_wrapper.tool_definition = registered
        async_wrapper.tool_name = tool_name
        return async_wrapper

    return decorator
