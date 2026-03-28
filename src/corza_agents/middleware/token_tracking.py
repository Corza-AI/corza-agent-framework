"""
Corza Agent Framework — Token Tracking Middleware

Tracks token usage and estimated cost per session.
Updates session totals in DB after each LLM call.
"""

import structlog

from corza_agents.core.types import (
    ExecutionContext,
    LLMResponse,
)
from corza_agents.middleware.base import BaseMiddleware
from corza_agents.persistence.base import BaseRepository

log = structlog.get_logger("corza_agents.token_tracking")

# Cost per million tokens — populate with your provider's pricing.
# Local models (Ollama) are free. Cloud providers vary.
COST_TABLE: dict[str, dict[str, float]] = {
    # Example entries (add your own):
    # "openai:gpt-5.4": {"input": 2.50, "output": 10.0},
    # "anthropic:claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}

DEFAULT_COST = {"input": 1.0, "output": 5.0}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given LLM call."""
    rates = COST_TABLE.get(model, DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


class TokenTrackingMiddleware(BaseMiddleware):
    """
    Tracks cumulative token usage and cost for each session.
    Updates the session record in DB after each LLM call.
    """

    def __init__(self, repository: BaseRepository):
        self._repo = repository
        self._session_totals: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "TokenTrackingMiddleware"

    async def after_llm_call(
        self,
        response: LLMResponse,
        context: ExecutionContext,
    ) -> LLMResponse:
        sid = context.session_id
        if sid not in self._session_totals:
            self._session_totals[sid] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        totals = self._session_totals[sid]
        totals["input_tokens"] += response.usage.input_tokens
        totals["output_tokens"] += response.usage.output_tokens
        totals["cost_usd"] += estimate_cost(
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        await self._repo.update_session(
            sid,
            total_input_tokens=totals["input_tokens"],
            total_output_tokens=totals["output_tokens"],
            total_cost_usd=totals["cost_usd"],
        )

        log.debug(
            "token_usage",
            session_id=sid,
            turn=context.turn_number,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cumulative_input=totals["input_tokens"],
            cumulative_output=totals["output_tokens"],
            cumulative_cost=f"${totals['cost_usd']:.4f}",
        )

        return response
