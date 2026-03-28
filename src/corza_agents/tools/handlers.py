"""
Corza Agent Framework — Tool Handlers

Execution handlers for non-function tool types:
  - API: HTTP REST calls with vault-resolved auth
  - DB_QUERY: PostgreSQL queries with vault-resolved credentials
  - WORKFLOW: Trigger external workflows
  - CODE: Python code execution in sandboxed subprocess

Mirrors Sentinel tool_registry.py patterns but async and generalized.
"""

import asyncio
import json
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urljoin

import structlog

from corza_agents.core.errors import ToolExecutionError
from corza_agents.core.types import (
    ExecutionContext,
    RegisteredTool,
    ToolCall,
    ToolResult,
    ToolStatus,
    ToolType,
)

log = structlog.get_logger("corza_agents.tools.handlers")


# ══════════════════════════════════════════════════════════════════════
# API Tool Handler
# ══════════════════════════════════════════════════════════════════════


async def execute_api_tool(
    tool_def: RegisteredTool,
    args: dict[str, Any],
    context: ExecutionContext,
    vault_resolver: Any | None = None,
) -> Any:
    """
    Execute an API tool — makes an HTTP request.

    Config keys expected on tool_def.config:
      - base_url: str (or resolved from vault)
      - endpoint: str (with {param} placeholders)
      - method: GET|POST|PUT|PATCH|DELETE
      - headers: dict (optional, merged with auth headers)
      - body_template: dict (optional, merged with args)
      - pagination: dict (optional, auto-pagination config)
      - vault_id: str (optional, for auth resolution)
    """
    import aiohttp

    config = tool_def.config
    method = config.get("method", "GET").upper()
    base_url = config.get("base_url", "")
    endpoint = config.get("endpoint", "")
    headers = dict(config.get("headers", {}))
    body_template = config.get("body_template", {})
    vault_id = config.get("vault_id") or tool_def.config.get("vault_id")

    # Resolve vault credentials
    if vault_resolver and vault_id:
        vault_data = await _resolve_vault(vault_resolver, vault_id, context)
        if vault_data:
            base_url = base_url or vault_data.get("base_url", "")
            auth_headers = _build_auth_headers(vault_data)
            headers.update(auth_headers)

    # Build URL with path parameters
    url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    for param_name, param_value in args.items():
        url = url.replace(f"{{{param_name}}}", str(param_value))

    # Build query params and body
    query_params = {}
    body = dict(body_template)
    for p in tool_def.parameters:
        val = args.get(p.name)
        if val is None:
            continue
        if p.config.get("location") == "query" if hasattr(p, "config") else False:
            query_params[p.name] = val
        elif p.name not in url:
            body[p.name] = val

    # Make the request
    async with aiohttp.ClientSession() as session:
        request_kwargs: dict[str, Any] = {
            "url": url,
            "headers": headers,
            "params": query_params if query_params else None,
        }
        if method in ("POST", "PUT", "PATCH") and body:
            request_kwargs["json"] = body

        async with session.request(method, **request_kwargs) as resp:
            status = resp.status
            try:
                result = await resp.json()
            except Exception:
                result = await resp.text()

            if status >= 400:
                raise ToolExecutionError(
                    f"API call failed: {status} {resp.reason}",
                    tool_name=tool_def.name,
                )

            # Auto-pagination
            pagination_config = config.get("pagination")
            if pagination_config and isinstance(result, dict):
                result = await _handle_pagination(
                    session, method, request_kwargs, result, pagination_config, headers
                )

            return result


async def _handle_pagination(
    session,
    method,
    base_kwargs,
    first_page,
    pagination_config,
    headers,
    max_pages: int = 10,
) -> dict:
    """Handle API pagination — accumulate results across pages."""
    results_key = pagination_config.get("results_key", "results")
    next_key = pagination_config.get("next_key", "next")
    all_results = first_page.get(results_key, [])

    current = first_page
    for _ in range(max_pages - 1):
        next_url = current.get(next_key)
        if not next_url:
            break
        async with session.request(method, url=next_url, headers=headers) as resp:
            if resp.status >= 400:
                break
            current = await resp.json()
            page_results = current.get(results_key, [])
            if not page_results:
                break
            all_results.extend(page_results)

    first_page[results_key] = all_results
    first_page["_pagination"] = {"total_pages_fetched": len(all_results)}
    return first_page


def _build_auth_headers(vault_data: dict) -> dict[str, str]:
    """Build auth headers from vault-resolved credentials."""
    auth_type = vault_data.get("auth_type", "bearer")
    if auth_type == "bearer":
        token = vault_data.get("token") or vault_data.get("api_key", "")
        return {"Authorization": f"Bearer {token}"}
    elif auth_type == "api_key":
        header_name = vault_data.get("header_name", "X-API-Key")
        return {header_name: vault_data.get("api_key", "")}
    elif auth_type == "basic":
        import base64

        username = vault_data.get("username", "")
        password = vault_data.get("password", "")
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    return {}


async def _resolve_vault(
    vault_resolver: Any, vault_id: str, context: ExecutionContext
) -> dict | None:
    """Resolve vault credentials. Supports both sync and async resolvers."""
    try:
        if asyncio.iscoroutinefunction(vault_resolver):
            return await vault_resolver(vault_id, context)
        return await asyncio.to_thread(vault_resolver, vault_id, context)
    except Exception as e:
        log.warning("vault_resolution_failed", vault_id=vault_id, error=str(e)[:200])
        return None


# ══════════════════════════════════════════════════════════════════════
# DB Query Tool Handler
# ══════════════════════════════════════════════════════════════════════


async def execute_db_query_tool(
    tool_def: RegisteredTool,
    args: dict[str, Any],
    context: ExecutionContext,
    vault_resolver: Any | None = None,
) -> Any:
    """
    Execute a database query tool.

    Config keys:
      - sql_query: str (with :param placeholders)
      - db_type: "postgresql" | "mysql" (default: postgresql)
      - vault_id: str (for DB credentials)
      - max_rows: int (default: 1000)
      - read_only: bool (default: True)
    """
    config = tool_def.config
    sql_query = config.get("sql_query", "")
    if not sql_query:
        raise ToolExecutionError("No sql_query in tool config", tool_name=tool_def.name)

    db_type = config.get("db_type", "postgresql")
    max_rows = config.get("max_rows", 1000)
    read_only = config.get("read_only", True)
    vault_id = config.get("vault_id")

    # Resolve DB credentials from vault
    db_url = None
    if vault_resolver and vault_id:
        vault_data = await _resolve_vault(vault_resolver, vault_id, context)
        if vault_data:
            db_url = vault_data.get("connection_string") or _build_db_url(vault_data, db_type)

    if not db_url:
        raise ToolExecutionError(
            f"No DB connection available for vault_id={vault_id}",
            tool_name=tool_def.name,
        )

    # Execute query
    return await asyncio.to_thread(_run_sql_query, db_url, sql_query, args, max_rows, read_only)


def _build_db_url(vault_data: dict, db_type: str) -> str:
    host = vault_data.get("host", "localhost")
    port = vault_data.get("port", 5432)
    database = vault_data.get("database", "")
    username = vault_data.get("username", "")
    password = vault_data.get("password", "")
    return f"{db_type}://{username}:{password}@{host}:{port}/{database}"


def _run_sql_query(db_url: str, sql: str, params: dict, max_rows: int, read_only: bool) -> dict:
    """Execute SQL query synchronously (runs in thread pool)."""
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            if read_only:
                conn = conn.execution_options(postgresql_readonly=True)
            result = conn.execute(text(sql), params)

            if result.returns_rows:
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchmany(max_rows)]
                return {
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": len(rows) >= max_rows,
                }
            else:
                return {"affected_rows": result.rowcount}
    finally:
        engine.dispose()


# ══════════════════════════════════════════════════════════════════════
# Workflow Tool Handler
# ══════════════════════════════════════════════════════════════════════


async def execute_workflow_tool(
    tool_def: RegisteredTool,
    args: dict[str, Any],
    context: ExecutionContext,
) -> Any:
    """
    Trigger a workflow execution.

    Config keys:
      - workflow_id: str
      - input_mapping: dict (maps tool params to workflow input events)
      - api_base_url: str (API URL, default from context)
      - wait_for_completion: bool (default: False)
      - timeout_seconds: int (default: 120)
    """
    config = tool_def.config
    workflow_id = config.get("workflow_id")
    if not workflow_id:
        raise ToolExecutionError("No workflow_id in tool config", tool_name=tool_def.name)

    input_mapping = config.get("input_mapping", {})
    api_base_url = config.get("api_base_url") or context.metadata.get(
        "workflow_api_url", "http://127.0.0.1:2222"
    )
    wait = config.get("wait_for_completion", False)
    timeout = config.get("timeout_seconds", 120)

    # Build workflow input
    workflow_input = {}
    for tool_param, workflow_field in input_mapping.items():
        if tool_param in args:
            workflow_input[workflow_field] = args[tool_param]

    # If no mapping, pass all args as workflow input
    if not input_mapping:
        workflow_input = args

    import aiohttp

    async with aiohttp.ClientSession() as session:
        trigger_url = f"{api_base_url.rstrip('/')}/api/v1/workflows/{workflow_id}/execute"
        async with session.post(trigger_url, json={"input": workflow_input}) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise ToolExecutionError(
                    f"Workflow trigger failed: {resp.status} {text[:500]}",
                    tool_name=tool_def.name,
                )
            result = await resp.json()

        if not wait:
            return {
                "status": "triggered",
                "execution_id": result.get("id"),
                "workflow_id": workflow_id,
            }

        # Poll for completion
        execution_id = result.get("id")
        status_url = f"{api_base_url.rstrip('/')}/api/v1/workflows/executions/{execution_id}"
        start = time.time()

        while time.time() - start < timeout:
            await asyncio.sleep(2)
            async with session.get(status_url) as resp:
                if resp.status >= 400:
                    continue
                status_data = await resp.json()
                status = status_data.get("status", "")
                if status in ("completed", "failed", "cancelled"):
                    return status_data

        return {"status": "timeout", "execution_id": execution_id, "workflow_id": workflow_id}


# ══════════════════════════════════════════════════════════════════════
# Code Execution Tool Handler
# ══════════════════════════════════════════════════════════════════════


async def execute_code_tool(
    tool_def: RegisteredTool,
    args: dict[str, Any],
    context: ExecutionContext,
) -> Any:
    """
    Execute Python code in a sandboxed subprocess.

    Config keys:
      - instructions: str (what the code should do — fed to LLM for code gen)
      - code: str (pre-defined code, if not using LLM generation)
      - sandbox_mode: "subprocess" | "docker" (default: subprocess)
      - timeout: int (default: 30)

    When 'code' is provided directly, it's executed in a subprocess.
    The code receives tool args as a JSON string on stdin and must print
    JSON results to stdout.
    """
    config = tool_def.config
    code = args.get("code") or config.get("code", "")
    timeout = config.get("timeout", 30)
    sandbox_mode = config.get("sandbox_mode", "subprocess")

    if not code:
        return {"status": "error", "message": "No code provided for execution"}

    # Safety gate: require explicit opt-in via environment variable
    import os

    if os.environ.get("CORZA_ALLOW_CODE_EXECUTION", "").lower() not in ("true", "1", "yes"):
        return {
            "status": "error",
            "message": (
                "Code execution is disabled. Set environment variable "
                "CORZA_ALLOW_CODE_EXECUTION=true to enable it. "
                "WARNING: This runs arbitrary Python code in a subprocess "
                "with no sandboxing. Only enable in trusted environments."
            ),
        }

    log.warning(
        "code_execution",
        tool_name=tool_def.name,
        code_length=len(code),
        timeout=timeout,
        sandbox_mode=sandbox_mode,
    )

    # Build input data for the sandbox
    input_data = json.dumps(args, default=str)

    if sandbox_mode == "subprocess":
        return await _run_in_subprocess(code, input_data, timeout)
    elif sandbox_mode == "docker":
        return await _run_in_docker(code, input_data, timeout)
    else:
        raise ToolExecutionError(f"Unknown sandbox_mode: {sandbox_mode}", tool_name=tool_def.name)


async def _run_in_subprocess(code: str, input_data: str, timeout: int) -> dict:
    """Run Python code in a subprocess sandbox."""
    wrapper = f"""
import sys, json

input_data = json.loads(sys.stdin.read())

{code}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            tmp_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_data.encode()),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            return {"status": "timeout", "message": f"Code execution timed out after {timeout}s"}

        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()

        if proc.returncode != 0:
            return {
                "status": "error",
                "exit_code": proc.returncode,
                "stderr": stderr_str[:3000],
                "stdout": stdout_str[:1000],
            }

        # Try to parse stdout as JSON
        try:
            result = json.loads(stdout_str)
        except json.JSONDecodeError:
            result = {"output": stdout_str[:5000]}

        return {"status": "success", "result": result}

    finally:
        import os

        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _run_in_docker(code: str, input_data: str, timeout: int) -> dict:
    """Run Python code in a Docker container (placeholder)."""
    return {
        "status": "error",
        "message": "Docker sandbox not yet implemented. Use sandbox_mode='subprocess'.",
    }


# ══════════════════════════════════════════════════════════════════════
# Handler Dispatcher
# ══════════════════════════════════════════════════════════════════════

TOOL_TYPE_HANDLERS = {
    ToolType.API: execute_api_tool,
    ToolType.DB_QUERY: execute_db_query_tool,
    ToolType.WORKFLOW: execute_workflow_tool,
    ToolType.CODE: execute_code_tool,
}


async def dispatch_tool(
    tool_def: RegisteredTool,
    tool_call: ToolCall,
    context: ExecutionContext,
    vault_resolver: Any | None = None,
) -> ToolResult:
    """
    Dispatch a tool call to the appropriate handler based on tool_type.

    For FUNCTION tools, the handler is called directly (via ToolRegistry).
    For other types, the appropriate handler from this module is used.
    """
    handler = TOOL_TYPE_HANDLERS.get(tool_def.tool_type)
    if not handler:
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            status=ToolStatus.ERROR,
            error=f"No handler for tool type: {tool_def.tool_type.value}",
        )

    start = time.time()
    try:
        kwargs: dict[str, Any] = {
            "tool_def": tool_def,
            "args": tool_call.arguments,
            "context": context,
        }
        if tool_def.tool_type in (ToolType.API, ToolType.DB_QUERY) and vault_resolver:
            kwargs["vault_resolver"] = vault_resolver

        result = await handler(**kwargs)
        duration = (time.time() - start) * 1000

        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            output=result,
            status=ToolStatus.SUCCESS,
            duration_ms=duration,
        )

    except Exception as e:
        duration = (time.time() - start) * 1000
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            status=ToolStatus.ERROR,
            duration_ms=duration,
            error=str(e)[:2000],
        )
