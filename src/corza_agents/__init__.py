"""
Corza Agent Framework
=====================

The agent framework built for web applications.

Drop into your FastAPI/Starlette app. Streams over SSE. Persists to
PostgreSQL. Knows about users, tenants, and sessions. Supports 23+
LLM providers — no default model, you choose at runtime.

Core concepts:
- Orchestrator: Brain agent that plans, delegates, and synthesizes
- AgentEngine: The ReAct loop (model → tools → model → ...)
- AgentDefinition: Declarative agent configuration (no default model)
- ToolRegistry: Central tool store with @tool decorator
- Skills: Reusable prompt template + tools bundles
- WorkingMemory: In-flight scratch space per session
- Middleware: Hook into the agent loop (audit, permissions, token tracking)
- Repository: Async DB persistence (PostgreSQL, SQLite, in-memory)
- StreamEvent: Real-time SSE events with reconnection support
- UserContext: Multi-tenant user/org context from HTTP headers

Quick start::

    from corza_agents import create_app, AgentDefinition, tool, ToolRegistry

    @tool(name="search", description="Search the database")
    async def search(query: str) -> dict:
        return {"results": [...]}

    tools = ToolRegistry()
    tools.register_function(search)

    assistant = AgentDefinition(
        name="assistant",
        model="openai:gpt-4.1",  # user provides model — no default
        tools=["search"],
    )

    app = create_app(
        agents={"assistant": assistant},
        db_url="postgresql://...",
        tool_registry=tools,
    )
    # Run with: uvicorn myapp:app

    # Or use programmatically:
    from corza_agents import Orchestrator, AgentLLM, create_repository

    llm = AgentLLM(api_keys={"openai": "sk-..."})
    repo = create_repository("postgres", db_url="postgresql+asyncpg://...")
    orchestrator = Orchestrator(llm, tools, repo)
    await orchestrator.initialize()

    async for event in orchestrator.run("session-1", "Analyze this", assistant):
        print(event.to_sse())  # SSE-formatted for streaming to frontend
"""

__version__ = "0.3.0"

from corza_agents.core.engine import AgentEngine
from corza_agents.core.errors import (
    AgentFrameworkError,
    ContextOverflowError,
    LLMError,
    LLMRateLimitError,
    MaxTurnsExceededError,
    MiddlewareError,
    SessionError,
    SessionNotFoundError,
    SkillNotFoundError,
    SubAgentError,
    ToolDeniedError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from corza_agents.core.llm import AgentLLM
from corza_agents.core.types import (
    AgentDefinition,
    AgentMessage,
    AgentSession,
    EventType,
    ExecutionContext,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
    MessageRole,
    RegisteredTool,
    SessionStatus,
    Skill,
    StopReason,
    SubAgentResult,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolStatus,
    ToolType,
    UserContext,
)
from corza_agents.memory.context import ContextManager
from corza_agents.memory.health import ContextHealthConfig, assess_health, compute_health_score
from corza_agents.memory.working import WorkingMemory
from corza_agents.middleware.audit import AuditMiddleware
from corza_agents.middleware.base import BaseMiddleware, Middleware
from corza_agents.middleware.context_compression import (
    CompressionConfig,
    ContextCompressionMiddleware,
    progressive_compress,
)
from corza_agents.middleware.loop_guard import LoopGuardConfig, LoopGuardMiddleware
from corza_agents.middleware.permissions import PermissionMiddleware, PermissionRule
from corza_agents.middleware.rate_limit import RateLimitMiddleware
from corza_agents.middleware.token_tracking import TokenTrackingMiddleware
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.orchestrator.sub_agent import SubAgentRunner
from corza_agents.persistence.base import BaseRepository
from corza_agents.persistence.factory import create_repository
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.persistence.models import Base

try:
    from corza_agents.persistence.repository import PostgresRepository, Repository
except ImportError:

    def _postgres_not_installed(*args, **kwargs):
        raise ImportError(
            "PostgresRepository requires asyncpg and sqlalchemy. "
            "Install with: pip install 'corza-agents[postgres]'"
        )

    PostgresRepository = _postgres_not_installed  # type: ignore[assignment,misc]
    Repository = _postgres_not_installed  # type: ignore[assignment,misc]

try:
    from corza_agents.persistence.sqlite import SQLiteRepository
except ImportError:

    def _sqlite_not_installed(*args, **kwargs):
        raise ImportError(
            "SQLiteRepository requires aiosqlite. Install with: pip install 'corza-agents[sqlite]'"
        )

    SQLiteRepository = _sqlite_not_installed  # type: ignore[assignment,misc]
from corza_agents.prompts.templates import (
    DEFAULT_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    TASK_AGENT_SYSTEM_PROMPT,
    build_system_prompt,
    load_knowledge,
)
from corza_agents.skills.manager import SkillsManager
from corza_agents.streaming.events import StreamEvent
from corza_agents.tools.builtin import BUILTIN_TOOLS
from corza_agents.tools.decorators import tool
from corza_agents.tools.handlers import TOOL_TYPE_HANDLERS, dispatch_tool
from corza_agents.tools.registry import ToolRegistry

try:
    from corza_agents.scheduler.scheduler import AgentScheduler, ScheduleEntry
except ImportError:
    AgentScheduler = None  # type: ignore[assignment,misc]
    ScheduleEntry = None  # type: ignore[assignment,misc]

try:
    from corza_agents.api.router import create_agent_router
    from corza_agents.api.service import AgentService
    from corza_agents.app import create_app
    from corza_agents.dependencies import get_service, get_user_context
except ImportError:
    create_agent_router = None  # type: ignore[assignment,misc]
    AgentService = None  # type: ignore[assignment,misc]
    create_app = None  # type: ignore[assignment,misc]
    get_service = None  # type: ignore[assignment,misc]
    get_user_context = None  # type: ignore[assignment,misc]

__all__ = [
    # Core types
    "AgentSession",
    "AgentMessage",
    "AgentDefinition",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
    "RegisteredTool",
    "Skill",
    "ExecutionContext",
    "SubAgentResult",
    "UserContext",
    "LLMStreamChunk",
    "LLMResponse",
    "LLMUsage",
    "SessionStatus",
    "MessageRole",
    "ToolStatus",
    "ToolType",
    "EventType",
    "StopReason",
    # Engine
    "AgentLLM",
    "AgentEngine",
    # Orchestrator
    "Orchestrator",
    "SubAgentRunner",
    # Tools
    "ToolRegistry",
    "tool",
    "BUILTIN_TOOLS",
    "dispatch_tool",
    "TOOL_TYPE_HANDLERS",
    # Skills
    "SkillsManager",
    # Memory
    "WorkingMemory",
    "ContextManager",
    # Middleware
    "Middleware",
    "BaseMiddleware",
    "AuditMiddleware",
    "TokenTrackingMiddleware",
    "PermissionMiddleware",
    "PermissionRule",
    "RateLimitMiddleware",
    "ContextCompressionMiddleware",
    "CompressionConfig",
    "progressive_compress",
    "LoopGuardMiddleware",
    "LoopGuardConfig",
    # Context Health
    "ContextHealthConfig",
    "assess_health",
    "compute_health_score",
    # Persistence
    "BaseRepository",
    "create_repository",
    "InMemoryRepository",
    "PostgresRepository",
    "SQLiteRepository",
    "Repository",
    "Base",
    # Streaming
    "StreamEvent",
    # Prompts
    "build_system_prompt",
    "load_knowledge",
    "DEFAULT_SYSTEM_PROMPT",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "TASK_AGENT_SYSTEM_PROMPT",
    # Scheduler
    "AgentScheduler",
    "ScheduleEntry",
    # API & Web
    "create_app",
    "create_agent_router",
    "AgentService",
    "get_service",
    "get_user_context",
    # Errors
    "AgentFrameworkError",
    "LLMError",
    "LLMRateLimitError",
    "ContextOverflowError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolDeniedError",
    "ToolTimeoutError",
    "SessionError",
    "SessionNotFoundError",
    "MaxTurnsExceededError",
    "SubAgentError",
    "MiddlewareError",
    "SkillNotFoundError",
]
