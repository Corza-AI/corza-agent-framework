"""Tests for built-in agent tools (objective, plan, notes, knowledge, skills)."""
import pytest

from corza_agents.core.types import ExecutionContext
from corza_agents.memory.working import WorkingMemory
from corza_agents.persistence.memory import InMemoryRepository
from corza_agents.tools.builtin import (
    manage_knowledge,
    manage_notes,
    manage_objective,
    manage_plan,
    manage_skill,
)


@pytest.fixture
def repo():
    return InMemoryRepository()


@pytest.fixture
def ctx(repo):
    """Create an ExecutionContext with working memory and repository."""
    wm = WorkingMemory()
    return ExecutionContext(
        session_id="test-session",
        agent_id="test-agent",
        turn_number=1,
        working_memory=wm,
        repository=repo,
    )


# ══════════════════════════════════════════════════════════════════════
# manage_objective
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_objective_read_empty(ctx):
    result = await manage_objective(action="read", ctx=ctx)
    assert result["status"] == "success"
    assert result["content"] is None
    assert "No objective" in result["message"]


@pytest.mark.asyncio
async def test_objective_write_and_read(ctx):
    result = await manage_objective(
        action="write",
        content="Analyze customer churn and identify retention strategies.",
        ctx=ctx,
    )
    assert result["status"] == "success"
    assert result["action"] == "written"

    result = await manage_objective(action="read", ctx=ctx)
    assert result["status"] == "success"
    assert "customer churn" in result["content"]


@pytest.mark.asyncio
async def test_objective_append(ctx):
    await manage_objective(action="write", content="Mission: reduce churn.", ctx=ctx)
    result = await manage_objective(
        action="append", content="Focus: Enterprise segment.", ctx=ctx
    )
    assert result["status"] == "success"
    assert result["action"] == "appended"

    result = await manage_objective(action="read", ctx=ctx)
    assert "Mission: reduce churn." in result["content"]
    assert "Focus: Enterprise segment." in result["content"]


@pytest.mark.asyncio
async def test_objective_write_empty_fails(ctx):
    result = await manage_objective(action="write", content="", ctx=ctx)
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_objective_invalid_action(ctx):
    result = await manage_objective(action="delete", ctx=ctx)
    assert result["status"] == "error"


# ══════════════════════════════════════════════════════════════════════
# manage_plan
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_plan_add_and_list(ctx):
    await manage_plan(action="add", item="Step 1: research", ctx=ctx)
    await manage_plan(action="add", item="Step 2: analyze", ctx=ctx)
    result = await manage_plan(action="list", ctx=ctx)
    assert result["count"] == 2
    assert result["plan"][0]["item"] == "Step 1: research"
    assert result["plan"][1]["status"] == "pending"


@pytest.mark.asyncio
async def test_plan_complete(ctx):
    await manage_plan(action="add", item="Do thing", ctx=ctx)
    result = await manage_plan(action="complete", item_id="1", ctx=ctx)
    assert result["entry"]["status"] == "done"


@pytest.mark.asyncio
async def test_plan_clear(ctx):
    await manage_plan(action="add", item="A", ctx=ctx)
    await manage_plan(action="add", item="B", ctx=ctx)
    await manage_plan(action="add", item="C", ctx=ctx)
    result = await manage_plan(action="clear", ctx=ctx)
    assert result["action"] == "cleared"
    assert result["items_removed"] == 3

    result = await manage_plan(action="list", ctx=ctx)
    assert result["plan"] == []


@pytest.mark.asyncio
async def test_plan_remove(ctx):
    await manage_plan(action="add", item="Keep", ctx=ctx)
    await manage_plan(action="add", item="Remove me", ctx=ctx)
    result = await manage_plan(action="remove", item_id="2", ctx=ctx)
    assert result["removed"]["item"] == "Remove me"

    result = await manage_plan(action="list", ctx=ctx)
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_plan_update_status(ctx):
    await manage_plan(action="add", item="Investigate", ctx=ctx)
    result = await manage_plan(
        action="update", item_id="1", status="in_progress", ctx=ctx
    )
    assert result["entry"]["status"] == "in_progress"


# ══════════════════════════════════════════════════════════════════════
# manage_notes
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_notes_read_empty(ctx):
    result = await manage_notes(action="read", ctx=ctx)
    assert result["notes"] == ""


@pytest.mark.asyncio
async def test_notes_write_and_read(ctx):
    await manage_notes(action="write", content="Some scratch notes", ctx=ctx)
    result = await manage_notes(action="read", ctx=ctx)
    assert result["notes"] == "Some scratch notes"


@pytest.mark.asyncio
async def test_notes_append(ctx):
    await manage_notes(action="write", content="Line 1", ctx=ctx)
    await manage_notes(action="append", content="Line 2", ctx=ctx)
    result = await manage_notes(action="read", ctx=ctx)
    assert "Line 1" in result["notes"]
    assert "Line 2" in result["notes"]


@pytest.mark.asyncio
async def test_notes_clear(ctx):
    await manage_notes(action="write", content="Something", ctx=ctx)
    await manage_notes(action="clear", ctx=ctx)
    result = await manage_notes(action="read", ctx=ctx)
    assert result["notes"] == ""


# ══════════════════════════════════════════════════════════════════════
# manage_knowledge
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_knowledge_write_and_read(ctx):
    result = await manage_knowledge(
        action="write", name="findings", content="Revenue grew 40%", ctx=ctx
    )
    assert result["status"] == "success"

    result = await manage_knowledge(action="read", name="findings", ctx=ctx)
    assert result["content"] == "Revenue grew 40%"


@pytest.mark.asyncio
async def test_knowledge_append(ctx):
    await manage_knowledge(
        action="write", name="facts", content="Fact 1.", ctx=ctx
    )
    await manage_knowledge(
        action="append", name="facts", content="Fact 2.", ctx=ctx
    )
    result = await manage_knowledge(action="read", name="facts", ctx=ctx)
    assert "Fact 1." in result["content"]
    assert "Fact 2." in result["content"]


@pytest.mark.asyncio
async def test_knowledge_list(ctx):
    await manage_knowledge(action="write", name="doc-a", content="AAA", ctx=ctx)
    await manage_knowledge(action="write", name="doc-b", content="BBBB", ctx=ctx)
    result = await manage_knowledge(action="list", ctx=ctx)
    assert result["count"] == 2
    names = [d["name"] for d in result["documents"]]
    assert "doc-a" in names
    assert "doc-b" in names


@pytest.mark.asyncio
async def test_knowledge_delete(ctx):
    await manage_knowledge(action="write", name="temp", content="X", ctx=ctx)
    result = await manage_knowledge(action="delete", name="temp", ctx=ctx)
    assert result["action"] == "deleted"

    result = await manage_knowledge(action="read", name="temp", ctx=ctx)
    assert result["content"] is None


@pytest.mark.asyncio
async def test_knowledge_read_nonexistent(ctx):
    result = await manage_knowledge(action="read", name="nope", ctx=ctx)
    assert result["content"] is None


# ══════════════════════════════════════════════════════════════════════
# manage_skill
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_skill_write_and_read(ctx):
    result = await manage_skill(
        action="write",
        name="data-validation",
        description="Validate data quality",
        content="Step 1: Check nulls\nStep 2: Check types",
        ctx=ctx,
    )
    assert result["status"] == "success"

    result = await manage_skill(action="read", name="data-validation", ctx=ctx)
    assert result["description"] == "Validate data quality"
    assert "Step 1" in result["content"]


@pytest.mark.asyncio
async def test_skill_list(ctx):
    await manage_skill(
        action="write", name="skill-a", description="Skill A",
        content="Do A", ctx=ctx,
    )
    await manage_skill(
        action="write", name="skill-b", description="Skill B",
        content="Do B", ctx=ctx,
    )
    result = await manage_skill(action="list", ctx=ctx)
    assert result["count"] == 2
    names = [s["name"] for s in result["skills"]]
    assert "skill-a" in names


@pytest.mark.asyncio
async def test_skill_delete(ctx):
    await manage_skill(
        action="write", name="temp-skill", description="Temp",
        content="...", ctx=ctx,
    )
    result = await manage_skill(action="delete", name="temp-skill", ctx=ctx)
    assert result["action"] == "deleted"

    result = await manage_skill(action="read", name="temp-skill", ctx=ctx)
    assert result["content"] is None
