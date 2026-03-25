"""Tests for skills and knowledge loading."""

from corza_agents import Skill, SkillsManager
from corza_agents.core.types import AgentDefinition
from corza_agents.prompts.templates import build_system_prompt, load_knowledge


def test_skill_creation():
    skill = Skill(name="test", prompt_template="Do the thing")
    assert skill.name == "test"
    assert skill.prompt_template == "Do the thing"


def test_skill_from_markdown():
    md = "# My Skill\n\nStep 1: Do this\nStep 2: Do that"
    skill = SkillsManager.from_markdown("s1", "My Skill", md, required_tools=["tool_a"])
    assert skill.id == "s1"
    assert skill.name == "My Skill"
    assert "Step 1" in skill.prompt_template
    assert "tool_a" in skill.required_tools


def test_skill_from_function():
    def my_analysis():
        """Analyze the data carefully.

        Step 1: Load the data
        Step 2: Run analysis
        Step 3: Report findings
        """
        pass

    skill = SkillsManager.from_function(my_analysis)
    assert skill.name == "my_analysis"
    assert "Analyze the data" in skill.description
    assert "Step 1" in skill.prompt_template


def test_skill_from_dict():
    data = {
        "name": "review",
        "prompt_template": "Review the code",
        "required_tools": ["lint"],
    }
    skill = SkillsManager.from_dict(data)
    assert skill.name == "review"


def test_skills_manager_register_and_resolve():
    mgr = SkillsManager()
    skill = Skill(id="s1", name="My Skill", prompt_template="Do it")
    mgr.register(skill)

    resolved = mgr.resolve(["s1"])
    assert len(resolved) == 1
    assert resolved[0].name == "My Skill"

    # Also resolvable by name
    resolved = mgr.resolve(["My Skill"])
    assert len(resolved) == 1


def test_skills_manager_missing_skill():
    mgr = SkillsManager()
    resolved = mgr.resolve(["nonexistent"])
    assert len(resolved) == 0


def test_skills_manager_required_tools():
    mgr = SkillsManager()
    skills = [
        Skill(name="a", prompt_template="...", required_tools=["t1", "t2"]),
        Skill(name="b", prompt_template="...", required_tools=["t2", "t3"]),
    ]
    tools = mgr.get_required_tools(skills)
    assert set(tools) == {"t1", "t2", "t3"}


# ── Knowledge loading ─────────────────────────────────────────────


def test_load_knowledge_empty():
    assert load_knowledge([]) == ""


def test_load_knowledge_inline_dict():
    result = load_knowledge([{"title": "Schema", "content": "orders(id, name)"}])
    assert "Schema" in result
    assert "orders" in result


def test_load_knowledge_callable():
    result = load_knowledge([lambda: "Dynamic content"])
    assert "Dynamic content" in result


def test_load_knowledge_mixed_sources(tmp_path):
    # File source
    f = tmp_path / "test.md"
    f.write_text("# File knowledge\nSome content")

    result = load_knowledge([
        str(f),
        {"title": "Inline", "content": "Inline content"},
        lambda: "Callable content",
    ])
    assert "File knowledge" in result
    assert "Inline" in result
    assert "Callable content" in result


def test_load_knowledge_missing_file():
    # Should not raise, just skip
    result = load_knowledge(["/nonexistent/file.md"])
    assert result == ""


# ── System prompt building ────────────────────────────────────────


def test_build_system_prompt_basic():
    agent = AgentDefinition(name="test", system_prompt="You are helpful.")
    prompt = build_system_prompt(agent)
    assert "You are helpful." in prompt


def test_build_system_prompt_with_skills():
    """Skills are listed as name + description only (lazy-loaded via manage_skill)."""
    agent = AgentDefinition(name="test")
    skills = [Skill(name="review", version="1.0", description="Code review procedure",
                     prompt_template="Review the code step by step")]
    prompt = build_system_prompt(agent, skills=skills)
    assert "review" in prompt
    assert "Code review procedure" in prompt
    assert "Available Skills" in prompt
    # Full content is NOT injected — agent reads it via manage_skill
    assert "Review the code step by step" not in prompt


def test_build_system_prompt_with_knowledge():
    """Knowledge is no longer injected into prompt — agent reads via manage_knowledge."""
    agent = AgentDefinition(
        name="test",
        knowledge=[{"title": "Project", "content": "This is our project"}],
    )
    prompt = build_system_prompt(agent)
    # Knowledge content is NOT in the prompt — it's loaded on demand
    # The prompt should still have the base system prompt
    assert "Orchestrator" in prompt or "autonomous" in prompt
