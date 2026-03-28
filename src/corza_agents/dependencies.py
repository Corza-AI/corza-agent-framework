"""
Corza Agent Framework — FastAPI Dependency Injection

Standard FastAPI dependencies for injecting agent services into endpoints.

Usage:
    from corza_agents.dependencies import get_service, get_user_context

    @router.post("/chat")
    async def chat(
        service: AgentService = Depends(get_service),
        user: UserContext = Depends(get_user_context),
    ):
        session = await service.create_session("assistant", user.user_id, user.tenant_id)
"""

from starlette.requests import Request

from corza_agents.api.service import AgentService
from corza_agents.core.types import UserContext


def get_service(request: Request) -> AgentService:
    """
    FastAPI dependency: inject the AgentService singleton.

    The service is stored on app.state by create_app() or manually.
    """
    service = getattr(request.app.state, "agent_service", None)
    if service is None:
        raise RuntimeError(
            "AgentService not found on app.state. "
            "Use create_app() or set app.state.agent_service manually."
        )
    return service


def get_user_context(request: Request) -> UserContext:
    """
    FastAPI dependency: extract user/tenant context from request headers.

    Reads these headers (your app sets them via auth middleware):
    - X-User-ID
    - X-Tenant-ID
    - X-Org-ID

    Override this dependency to use JWT claims, session cookies, etc.
    """
    return UserContext(
        user_id=request.headers.get("X-User-ID", ""),
        tenant_id=request.headers.get("X-Tenant-ID", ""),
        org_id=request.headers.get("X-Org-ID", ""),
    )
