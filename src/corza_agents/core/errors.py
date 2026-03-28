"""
Corza Agent Framework — Error Types
"""


class AgentFrameworkError(Exception):
    """Base exception for all framework errors."""

    pass


class LLMError(AgentFrameworkError):
    """Error during LLM call."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        status_code: int = 0,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable


class LLMRateLimitError(LLMError):
    """Rate limit exceeded — retryable."""

    def __init__(self, message: str, retry_after_seconds: float = 0, **kwargs):
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after_seconds = retry_after_seconds


class ContextOverflowError(LLMError):
    """Context window exceeded."""

    def __init__(self, message: str, tokens_used: int = 0, token_limit: int = 0, **kwargs):
        super().__init__(message, **kwargs)
        self.tokens_used = tokens_used
        self.token_limit = token_limit


class ToolExecutionError(AgentFrameworkError):
    """Error during tool execution."""

    def __init__(
        self,
        message: str,
        tool_name: str = "",
        tool_call_id: str = "",
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.original_error = original_error


class ToolNotFoundError(ToolExecutionError):
    """Requested tool does not exist in registry."""

    pass


class ToolDeniedError(ToolExecutionError):
    """Tool execution was denied by permission middleware."""

    pass


class ToolTimeoutError(ToolExecutionError):
    """Tool execution timed out."""

    pass


class SessionError(AgentFrameworkError):
    """Error related to session management."""

    pass


class SessionNotFoundError(SessionError):
    """Session does not exist."""

    pass


class MaxTurnsExceededError(AgentFrameworkError):
    """Agent reached max_turns without completing."""

    def __init__(self, session_id: str, turns: int, max_turns: int):
        super().__init__(f"Session {session_id} reached max turns ({turns}/{max_turns})")
        self.session_id = session_id
        self.turns = turns
        self.max_turns = max_turns


class SubAgentError(AgentFrameworkError):
    """Error in a sub-agent execution."""

    def __init__(
        self, message: str, child_session_id: str = "", original_error: Exception | None = None
    ):
        super().__init__(message)
        self.child_session_id = child_session_id
        self.original_error = original_error


class MiddlewareError(AgentFrameworkError):
    """Error in middleware processing."""

    def __init__(self, message: str, middleware_name: str = ""):
        super().__init__(message)
        self.middleware_name = middleware_name


class SkillNotFoundError(AgentFrameworkError):
    """Requested skill does not exist."""

    pass
