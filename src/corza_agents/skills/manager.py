"""
Corza Agent Framework — Skills Manager

Skills are reusable procedures — step-by-step instructions that tell
an agent HOW to do a specific task. They pair a prompt template with
the tools needed to execute it.

System Prompt = Principles (who the agent is)
Skills        = Procedures (what to do)
"""
import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import structlog

from corza_agents.core.types import Skill
from corza_agents.persistence.base import BaseRepository

log = structlog.get_logger("corza_agents.skills")


class SkillsManager:
    """
    Manages skill definitions — loading, resolution, and prompt injection.

    Skills are composable prompt templates + tool requirements.
    They tell the agent HOW to do something specific.
    """

    def __init__(self, repository: BaseRepository | None = None):
        self._repository = repository
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill definition in memory."""
        self._skills[skill.id] = skill
        if skill.name and skill.name != skill.id:
            self._skills[skill.name] = skill
        log.debug("skill_registered", id=skill.id, name=skill.name)

    def register_many(self, skills: list[Skill]) -> None:
        for s in skills:
            self.register(s)

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def resolve(self, skill_ids: list[str]) -> list[Skill]:
        """
        Resolve a list of skill IDs/names to Skill objects.
        Logs warnings for missing skills but doesn't fail.
        """
        resolved = []
        for sid in skill_ids:
            skill = self._skills.get(sid)
            if skill:
                resolved.append(skill)
            else:
                log.warning("skill_not_found", skill_id=sid)
        return resolved

    def get_required_tools(self, skills: list[Skill]) -> list[str]:
        """Collect all required tool names from a set of skills."""
        tools: set[str] = set()
        for skill in skills:
            tools.update(skill.required_tools)
        return list(tools)

    @staticmethod
    def from_markdown(
        skill_id: str,
        name: str,
        markdown_content: str,
        required_tools: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Skill:
        """
        Create a Skill from a markdown file's content.
        Useful for loading .md playbooks as skills.
        """
        return Skill(
            id=skill_id,
            name=name,
            description=markdown_content.split("\n")[0].strip("# ").strip(),
            prompt_template=markdown_content,
            required_tools=required_tools or [],
            config=config or {},
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Skill:
        """Create a Skill from a dictionary (e.g., from DB or JSON config)."""
        return Skill(**data)

    @staticmethod
    def from_function(fn: Callable) -> Skill:
        """
        Create a Skill from a Python function.

        The function's docstring becomes the prompt template.
        The function name becomes the skill ID/name.
        Attach a `required_tools` list attribute to the function
        to specify tool dependencies.
        """
        doc = inspect.getdoc(fn) or ""
        name = getattr(fn, "__name__", "unnamed_skill")
        required_tools = getattr(fn, "required_tools", [])
        return Skill(
            id=name,
            name=name,
            description=doc.split("\n")[0] if doc else name,
            prompt_template=doc,
            required_tools=required_tools,
        )

    @staticmethod
    async def from_url(
        url: str,
        skill_id: str | None = None,
        name: str | None = None,
        required_tools: list[str] | None = None,
    ) -> Skill:
        """
        Load a skill from a URL at runtime (expects markdown content).

        Args:
            url: URL to fetch markdown from.
            skill_id: Override skill ID (defaults to filename from URL).
            name: Override skill name (defaults to skill_id).
            required_tools: Tools needed by this skill.
        """
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.text()
        _id = skill_id or url.split("/")[-1].replace(".md", "")
        _name = name or _id
        return SkillsManager.from_markdown(_id, _name, content, required_tools)

    @staticmethod
    async def from_database(
        query_fn: Callable,
        skill_id: str | None = None,
    ) -> Skill:
        """
        Load a skill from a database query callback.

        The callback should return either:
        - A dict with keys: name, prompt_template, required_tools, config, description
        - A string (treated as the prompt_template)

        Supports both sync and async callbacks.
        """
        if asyncio.iscoroutinefunction(query_fn):
            result = await query_fn()
        else:
            result = query_fn()

        if isinstance(result, str):
            _id = skill_id or "db_skill"
            return Skill(id=_id, name=_id, prompt_template=result)
        elif isinstance(result, dict):
            return Skill(
                id=skill_id or result.get("id", "db_skill"),
                name=result.get("name", skill_id or "db_skill"),
                prompt_template=result.get("prompt_template", ""),
                required_tools=result.get("required_tools", []),
                config=result.get("config", {}),
                description=result.get("description", ""),
            )
        else:
            raise ValueError(
                f"query_fn must return str or dict, got {type(result).__name__}"
            )
