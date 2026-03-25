"""Tests for prompt assembly (build_system_prompt and helpers)."""

from corza_agents.core.types import AgentDefinition, ToolSchema
from corza_agents.prompts.templates import _format_plan, build_system_prompt

# ══════════════════════════════════════════════════════════════════════
# Section ordering
# ══════════════════════════════════════════════════════════════════════

def test_build_system_prompt_section_ordering():
    """Sections must appear in the documented order: Objective, Tools, Knowledge, Skills, Working Memory, Plan, Extra."""
    agent_def = AgentDefinition(
        name="test-agent",
        system_prompt="You are a test agent.",
    )

    prompt = build_system_prompt(
        agent_def,
        objective="test objective",
        knowledge_index=[{"name": "doc1", "size": 100}],
        skill_index=[{"name": "skill1", "description": "test"}],
        working_memory_context="working mem",
        plan=[{"id": "1", "item": "test", "status": "pending"}],
        registered_tools=[ToolSchema(name="tool1", description="test tool", parameters={})],
        extra_context="extra",
    )

    idx_objective = prompt.index("## Objective")
    idx_tools = prompt.index("## Available Tools")
    idx_knowledge = prompt.index("## Knowledge Library")
    idx_skills = prompt.index("## Available Skills")
    idx_memory = prompt.index("## Working Memory")
    idx_plan = prompt.index("## Active Plan")
    idx_extra = prompt.index("## Additional Context")

    assert idx_objective < idx_tools, "Objective must come before Available Tools"
    assert idx_tools < idx_knowledge, "Available Tools must come before Knowledge Library"
    assert idx_knowledge < idx_skills, "Knowledge Library must come before Available Skills"
    assert idx_skills < idx_memory, "Available Skills must come before Working Memory"
    assert idx_memory < idx_plan, "Working Memory must come before Active Plan"
    assert idx_plan < idx_extra, "Active Plan must come before Additional Context"


# ══════════════════════════════════════════════════════════════════════
# Turn count injection
# ══════════════════════════════════════════════════════════════════════

def test_build_system_prompt_turn_count():
    """When a plan is present, the turn counter should appear."""
    agent_def = AgentDefinition(name="counter-agent")

    prompt = build_system_prompt(
        agent_def,
        plan=[{"id": "1", "item": "task", "status": "pending"}],
        turn_number=3,
        max_turns=10,
    )

    assert "Turn 3 of 10" in prompt


def test_build_system_prompt_no_turn_count_without_plan():
    """Without a plan, the turn counter should NOT appear (it lives in the plan section)."""
    agent_def = AgentDefinition(name="no-plan-agent")

    prompt = build_system_prompt(
        agent_def,
        plan=None,
        turn_number=3,
        max_turns=10,
    )

    assert "Turn 3 of 10" not in prompt


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
