"""
Corza Agent Framework — Prompt Templates

System prompt construction, knowledge file loading, and skill injection.

Architecture:
    System Prompt = Principles (WHO the agent is, HOW it thinks)
    Knowledge     = Context    (WHAT the agent knows — loaded from files)
    Skills        = Procedures (WHAT to do for a specific task)

Keep system prompts short, principled, and domain-agnostic.
Put project context, schemas, and reference material into knowledge files.
Put step-by-step procedures, checklists, and domain logic into Skills.
"""

from pathlib import Path
from typing import Any

import structlog

from corza_agents.core.types import AgentDefinition, Skill

try:
    from jinja2 import Template as Jinja2Template

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

log = structlog.get_logger("corza_agents.prompts")


# ══════════════════════════════════════════════════════════════════════
# Orchestrator System Prompt
#
# This is the DEFAULT prompt. The framework always uses the
# Orchestrator + Sub-Agents architecture. The orchestrator owns
# strategy and delegation. Sub-agents own depth and execution.
# ══════════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator. You decompose problems, delegate depth to sub-agents, and synthesize results for the user.

## How You Operate

You think in tasks. When a problem arrives, break it into independent pieces, track them with `manage_plan`, and delegate each to the right sub-agent. Evaluate what comes back — push back on thin work. Synthesize the findings into clear insight for the user.

Not everything needs decomposition. Simple questions get direct answers. Use your judgment.

## Principles

- **Think out loud** — the user watches your process in real-time. Before every action, explain what you're about to do and why. After every result, interpret what it means. Never call tools back-to-back without narration in between.
- **Judgment over effort** — match your approach to the problem. Don't over-engineer simple requests.
- **Delegate depth** — sub-agents do the deep work. You own strategy, coordination, and synthesis. Give each agent a specific scope and clear success criteria.
- **Quality over quantity** — one thorough investigation beats five shallow ones. Be a skeptic with sub-agent results.
- **Build on what exists** — check knowledge and skills before starting new work. Don't redo what prior sessions already produced.
- **Learn and persist** — save findings to `manage_knowledge` and successful workflows to `manage_skill`.
- **Never fabricate** — "we couldn't determine X because Y" is valuable. Guessing is not.
- **Lead with insight** — deliver the most important finding first. Support with evidence. Note gaps.

Only call tools listed in your Tools section below.
"""

# Backward compatibility — DEFAULT is always the orchestrator prompt
DEFAULT_SYSTEM_PROMPT = ORCHESTRATOR_SYSTEM_PROMPT

# ══════════════════════════════════════════════════════════════════════
# Task Agent System Prompt — Investigation Principles
# ══════════════════════════════════════════════════════════════════════

TASK_AGENT_SYSTEM_PROMPT = """You are a Task Agent. You receive a task from the Orchestrator, execute it thoroughly, and report back. That is your entire lifecycle.

## How You Operate

You own depth on a single thread. If the task is complex, use `manage_plan` to break it into sub-steps and track your progress. Work through each step, record meaningful findings to `manage_knowledge` as you go, and report everything back via `manage_agent(action="report")`. The Orchestrator cannot see your tool calls or reasoning — only your reports reach them.

## Principles

- **Think out loud** — the user watches your process in real-time. Before every tool call, explain what you're about to do and why. After every result, interpret what it means. Never call tools back-to-back without narration in between.
- **Depth over breadth** — follow every lead to its root. If a result surprises you, that's a signal to dig deeper.
- **Build on what exists** — check knowledge and skills before querying. Prior sessions may have the answer.
- **Record as you go** — persist findings to `manage_knowledge` immediately. Each finding should be self-contained with specific evidence.
- **Never fabricate** — if you can't find the answer, say so. "I couldn't determine X because Y" is valuable. Guessing is not.

Only call tools listed in your Tools section below.
"""


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Render a template string with variables."""
    if HAS_JINJA2:
        return Jinja2Template(template).render(**variables)
    try:
        return template.format(**variables)
    except (KeyError, IndexError):
        return template


def load_knowledge(sources: list) -> str:
    """
    Load knowledge from multiple source types.

    Sources can be:
    - str: File path or glob pattern (e.g. "knowledge/*.md")
    - dict: Inline knowledge with "title" and "content" keys
    - Callable: Sync function that returns a string

    Files that don't exist are silently skipped with a warning.
    """
    if not sources:
        return ""

    sections: list[str] = []

    for source in sources:
        if callable(source) and not isinstance(source, str):
            # Callable knowledge source
            try:
                result = source()
                if result and isinstance(result, str):
                    sections.append(result)
            except Exception as e:
                log.warning("knowledge_callable_error", error=str(e))

        elif isinstance(source, dict):
            # Inline knowledge: {"title": "...", "content": "..."}
            title = source.get("title", "Knowledge")
            content = source.get("content", "")
            if content:
                sections.append(f"### {title}\n\n{content}")

        elif isinstance(source, str):
            # File path or glob pattern (original behavior)
            path = Path(source).expanduser()

            if "*" in source:
                base = Path(path.parts[0]) if path.is_absolute() else Path(".")
                pattern = str(path.relative_to(base)) if path.is_absolute() else source
                matched = sorted(base.glob(pattern))
                for match in matched:
                    content = _read_knowledge_file(match)
                    if content:
                        sections.append(content)
                continue

            if not path.is_absolute():
                path = Path.cwd() / path

            content = _read_knowledge_file(path)
            if content:
                sections.append(content)

    return "\n\n".join(sections)


def _read_knowledge_file(path: Path) -> str | None:
    """Read a single knowledge file. Returns None if not found."""
    try:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            log.debug("knowledge_loaded", path=str(path), chars=len(content))
            return content
        return None
    except FileNotFoundError:
        log.warning("knowledge_file_not_found", path=str(path))
        return None
    except Exception as e:
        log.warning("knowledge_file_error", path=str(path), error=str(e))
        return None


def _format_plan(plan: list[dict]) -> str:
    """Format a plan list into a readable string for prompt injection."""
    if not plan:
        return "Empty — no plan items yet."

    status_icons = {
        "pending": "○",
        "in_progress": "◉",
        "done": "✓",
        "blocked": "✗",
    }
    lines = []
    for item in plan:
        icon = status_icons.get(item.get("status", "pending"), "○")
        item_id = item.get("id", "?")
        text = item.get("item", "")
        status = item.get("status", "pending")
        lines.append(f"{icon} [{item_id}] {text} ({status})")
    return "\n".join(lines)


_OBJECTIVE_MAX_CHARS = 2000


def build_system_prompt(
    agent_def: AgentDefinition,
    skills: list[Skill] | None = None,
    working_memory_context: str = "",
    extra_context: str = "",
    variables: dict[str, Any] | None = None,
    registered_tools: list | None = None,
    knowledge_index: list[dict] | None = None,
    skill_index: list[dict] | None = None,
    objective: str | None = None,
    plan: list[dict] | None = None,
) -> str:
    """
    Construct the full system prompt for an agent.

    Structure:
        1. Identity   — WHO the agent is (one-liner)
        2. Principles — HOW it operates (modus operandi)
        3. Objective   — WHAT it's working on right now
        4. Skills      — available procedures (name + 1-liner)
        5. Knowledge   — available context (name + 1-liner)
        6. Tools       — available tools (name + 1-liner)
        7. Notes       — session scratch pad
        8. Plan        — active task tracker

    Identity and principles come from the agent's system_prompt.
    Everything else is assembled here from runtime state.
    """
    variables = variables or {}
    parts: list[str] = []

    # 1–2. Identity + Principles (from the agent's system prompt)
    base = agent_def.system_prompt or DEFAULT_SYSTEM_PROMPT
    parts.append(render_template(base, variables))

    # 3. Objective — the agent's current mission (capped to prevent bloat)
    obj_text = objective or agent_def.objective
    if obj_text:
        if len(obj_text) > _OBJECTIVE_MAX_CHARS:
            obj_text = (
                obj_text[:_OBJECTIVE_MAX_CHARS]
                + "\n\n*(truncated — load full objective via `manage_objective(action='read')`)*"
            )
        parts.append(f"## Objective\n\n{obj_text}")

    # 4. Skills — reusable procedures
    #    Code-defined skills with prompt_template: inject full content.
    #    DB skills (skill_index): show name + description index (load via manage_skill).
    if skills:
        # Registered Skill objects — inject full prompt_template if present
        for skill in skills:
            if skill.prompt_template:
                parts.append(
                    f"## Skill: {skill.name}\n\n{skill.prompt_template}"
                )
    if skill_index:
        # DB-stored skills — show index only (agent loads full content on demand)
        remaining = skill_index
        if skills:
            loaded_names = {s.name for s in skills}
            remaining = [sk for sk in skill_index if sk.get("name", "") not in loaded_names]
        if remaining:
            lines = ["## Skills\n"]
            for sk in remaining:
                name = sk.get("name", "")
                desc = sk.get("description", "")
                lines.append(f"- **{name}** — {desc}")
            parts.append("\n".join(lines))

    # 5. Knowledge — persistent context (name + description index)
    if knowledge_index is not None:
        if knowledge_index:
            lines = ["## Knowledge\n"]
            for doc in knowledge_index:
                name = doc.get("name", "")
                desc = doc.get("description", "")
                if desc:
                    lines.append(f"- **{name}** — {desc}")
                else:
                    lines.append(f"- **{name}**")
            parts.append("\n".join(lines))
        else:
            parts.append("## Knowledge\n\nNo documents yet.")

    # 6. Tools — available tool functions (name + description index)
    if registered_tools:
        lines = ["## Tools\n"]
        for t in registered_tools:
            name = t.name if hasattr(t, "name") else str(t)
            desc = t.description if hasattr(t, "description") else ""
            lines.append(f"- **{name}** — {desc}")
        parts.append("\n".join(lines))

    # 7. Notes — session scratch context
    if working_memory_context:
        parts.append(f"## Notes\n\n{working_memory_context}")

    # 8. Plan — active task tracker (near the end for recency bias)
    if plan is not None:
        parts.append(f"## Plan\n\n{_format_plan(plan)}")

    # Extra context (application-specific, appended last)
    if extra_context:
        parts.append(f"## Additional Context\n\n{extra_context}")

    return "\n\n".join(parts)
