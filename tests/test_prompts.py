"""Tests for prompt assembly (build_system_prompt and helpers)."""

from corza_agents.core.types import AgentDefinition, ToolSchema
from corza_agents.prompts.templates import _format_plan, build_system_prompt, _OBJECTIVE_MAX_CHARS

# ══════════════════════════════════════════════════════════════════════
# Section ordering: Identity → Principles → Objective → Skills →
#                   Knowledge → Tools → Notes → Plan
# ══════════════════════════════════════════════════════════════════════

def test_build_system_prompt_section_ordering():
    """Sections must appear in order: Objective, Skills, Knowledge, Tools, Notes, Plan, Extra."""
    agent_def = AgentDefinition(
        name="test-agent",
        system_prompt="You are a test agent.",
    )

    prompt = build_system_prompt(
        agent_def,
        objective="test objective",
        knowledge_index=[{"name": "doc1", "description": "Overview of database schema"}],
        skill_index=[{"name": "skill1", "description": "test"}],
        working_memory_context="working mem",
        plan=[{"id": "1", "item": "test", "status": "pending"}],
        registered_tools=[ToolSchema(name="tool1", description="test tool", parameters={})],
        extra_context="extra",
    )

    idx_objective = prompt.index("## Objective")
    idx_skills = prompt.index("## Skills")
    idx_knowledge = prompt.index("## Knowledge")
    idx_tools = prompt.index("## Tools")
    idx_notes = prompt.index("## Notes")
    idx_plan = prompt.index("## Plan")
    idx_extra = prompt.index("## Additional Context")

    assert idx_objective < idx_skills, "Objective must come before Skills"
    assert idx_skills < idx_knowledge, "Skills must come before Knowledge"
    assert idx_knowledge < idx_tools, "Knowledge must come before Tools"
    assert idx_tools < idx_notes, "Tools must come before Notes"
    assert idx_notes < idx_plan, "Notes must come before Plan"
    assert idx_plan < idx_extra, "Plan must come before Additional Context"


def test_build_system_prompt_no_reference_wrapper():
    """Skills and Knowledge should be top-level ## sections, not nested under ## Reference."""
    agent_def = AgentDefinition(name="test-agent")

    prompt = build_system_prompt(
        agent_def,
        knowledge_index=[{"name": "doc1", "description": "Overview of database schema"}],
        skill_index=[{"name": "skill1", "description": "test"}],
    )

    assert "## Reference" not in prompt, "No Reference wrapper section"
    assert "### Knowledge" not in prompt, "Knowledge is ## not ###"
    assert "### Skills" not in prompt, "Skills is ## not ###"
    assert "## Skills" in prompt
    assert "## Knowledge" in prompt


def test_build_system_prompt_no_framing_line():
    """No duplicate framing line between identity and context."""
    agent_def = AgentDefinition(name="test-agent")

    prompt = build_system_prompt(agent_def, objective="test")

    assert "not steps to follow" not in prompt


# ══════════════════════════════════════════════════════════════════════
# Objective truncation
# ══════════════════════════════════════════════════════════════════════

def test_build_system_prompt_objective_truncation():
    """Long objectives are truncated to prevent prompt bloat."""
    agent_def = AgentDefinition(name="test-agent")
    long_objective = "x" * (_OBJECTIVE_MAX_CHARS + 500)

    prompt = build_system_prompt(agent_def, objective=long_objective)

    assert "truncated" in prompt
    assert "manage_objective" in prompt


def test_build_system_prompt_short_objective_not_truncated():
    """Short objectives pass through unchanged."""
    agent_def = AgentDefinition(name="test-agent")
    short_objective = "Analyze customer churn."

    prompt = build_system_prompt(agent_def, objective=short_objective)

    assert "truncated" not in prompt
    assert short_objective in prompt


# ══════════════════════════════════════════════════════════════════════
# Plan is a task tracker, not a turn counter
# ══════════════════════════════════════════════════════════════════════

def test_build_system_prompt_plan_has_no_turn_counter():
    """Plan section should never include a turn counter — it's a task tracker, not a countdown."""
    agent_def = AgentDefinition(name="tracker-agent")

    prompt = build_system_prompt(
        agent_def,
        plan=[{"id": "1", "item": "task", "status": "pending"}],
    )

    assert "## Plan" in prompt
    plan_section = prompt.split("## Plan")[1]
    assert "(turn" not in plan_section


# ══════════════════════════════════════════════════════════════════════
# Plan formatting
# ══════════════════════════════════════════════════════════════════════

def test_format_plan():
    """_format_plan should render status icons for each state."""
    plan = [
        {"id": "1", "item": "Gather data", "status": "done"},
        {"id": "2", "item": "Analyze results", "status": "in_progress"},
        {"id": "3", "item": "Write report", "status": "pending"},
    ]

    output = _format_plan(plan)

    assert "✓" in output, "Done items should have a checkmark"
    assert "◉" in output, "In-progress items should have a filled circle"
    assert "○" in output, "Pending items should have an open circle"
    assert "Gather data" in output
    assert "Analyze results" in output
    assert "Write report" in output


def test_format_plan_empty():
    """An empty plan should produce a placeholder message."""
    output = _format_plan([])
    assert "Empty" in output
