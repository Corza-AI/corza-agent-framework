"""
Corza Agent Framework — Working Memory

In-flight scratch space for the current agent session.
Fast, in-process, per-session. NOT persisted (session memory handles that).

Inspired by Sentinel agent's memory.py (data cache, findings, red flags).
"""
from typing import Any

import structlog

log = structlog.get_logger("corza_agents.memory")


class WorkingMemory:
    """
    In-process working memory for an agent session.

    Stores intermediate data, tool results, findings, and metadata
    that the agent accumulates during execution. This is the fast
    scratchpad — session-scoped, not persisted to DB.

    The context manager pattern ensures working memory is available
    during the agent run and cleaned up after.
    """

    def __init__(self, session_id: str = "", metadata: dict | None = None):
        self.session_id = session_id
        self.metadata = metadata or {}
        self._data_cache: dict[str, Any] = {}
        self._findings: list[dict[str, Any]] = []
        self._artifacts: dict[str, Any] = {}
        self._counters: dict[str, int] = {}
        self._tags: set[str] = set()

    # ══════════════════════════════════════════════════════════════════
    # Data Cache — store and retrieve tool results, intermediate data
    # ══════════════════════════════════════════════════════════════════

    def store(self, key: str, value: Any) -> None:
        self._data_cache[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data_cache.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data_cache

    def remove(self, key: str) -> None:
        self._data_cache.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._data_cache.keys())

    @property
    def data_cache(self) -> dict[str, Any]:
        return dict(self._data_cache)

    # ══════════════════════════════════════════════════════════════════
    # Findings — structured observations accumulated during the run
    # ══════════════════════════════════════════════════════════════════

    def add_finding(self, description: str, category: str = "",
                    evidence: dict | None = None, **kwargs) -> None:
        self._findings.append({
            "description": description,
            "category": category,
            "evidence": evidence or {},
            **kwargs,
        })

    @property
    def findings(self) -> list[dict[str, Any]]:
        return list(self._findings)

    def findings_by_category(self, category: str) -> list[dict]:
        return [f for f in self._findings if f.get("category") == category]

    # ══════════════════════════════════════════════════════════════════
    # Artifacts — named outputs generated during the session
    # ══════════════════════════════════════════════════════════════════

    def set_artifact(self, name: str, value: Any) -> None:
        self._artifacts[name] = value

    def get_artifact(self, name: str) -> Any | None:
        return self._artifacts.get(name)

    @property
    def artifacts(self) -> dict[str, Any]:
        return dict(self._artifacts)

    # ══════════════════════════════════════════════════════════════════
    # Counters — track occurrences
    # ══════════════════════════════════════════════════════════════════

    def increment(self, key: str, amount: int = 1) -> int:
        self._counters[key] = self._counters.get(key, 0) + amount
        return self._counters[key]

    def get_count(self, key: str) -> int:
        return self._counters.get(key, 0)

    # ══════════════════════════════════════════════════════════════════
    # Tags — simple labels for the session
    # ══════════════════════════════════════════════════════════════════

    def tag(self, *tags: str) -> None:
        self._tags.update(tags)

    def has_tag(self, tag: str) -> bool:
        return tag in self._tags

    @property
    def tags(self) -> set[str]:
        return set(self._tags)

    # ══════════════════════════════════════════════════════════════════
    # Context serialization — for LLM prompt injection
    # ══════════════════════════════════════════════════════════════════

    def get_context_for_llm(self, max_chars: int = 8000) -> str:
        """
        Serialize working memory into a compact string for LLM context window.
        Prioritizes findings and recent data over raw cache.
        """
        parts: list[str] = []

        if self._findings:
            parts.append("### Findings So Far")
            for i, f in enumerate(self._findings[-20:], 1):
                cat = f" [{f['category']}]" if f.get("category") else ""
                parts.append(f"{i}. {f['description']}{cat}")

        if self._data_cache:
            parts.append("\n### Available Data")
            for key in list(self._data_cache.keys())[-15:]:
                val = self._data_cache[key]
                preview = self._preview(val, max_length=300)
                parts.append(f"- **{key}**: {preview}")

        if self._tags:
            parts.append(f"\n### Tags: {', '.join(sorted(self._tags))}")

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars - 50] + "\n\n... (working memory truncated)"
        return text

    @staticmethod
    def _preview(value: Any, max_length: int = 300) -> str:
        if isinstance(value, str):
            return value[:max_length] + ("..." if len(value) > max_length else "")
        if isinstance(value, dict):
            keys = list(value.keys())[:5]
            return f"dict with keys: {keys}" + (f" (+{len(value) - 5} more)" if len(value) > 5 else "")
        if isinstance(value, list):
            return f"list with {len(value)} items"
        return str(value)[:max_length]

    def clear(self) -> None:
        self._data_cache.clear()
        self._findings.clear()
        self._artifacts.clear()
        self._counters.clear()
        self._tags.clear()
