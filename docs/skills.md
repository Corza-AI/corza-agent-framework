# Principles, Knowledge, and Skills

This is the core design idea behind the framework.

## The Problem

Most agent frameworks stuff everything into one giant system prompt — identity, project context, procedures, domain knowledge, guardrails. It becomes a monolithic blob that's hard to maintain, hard to reuse, and wastes tokens when half of it isn't relevant.

## The Solution

Split it into three layers:

```
┌─────────────────────────────────────────────┐
│  System Prompt (Principles)                 │  WHO the agent is, HOW it thinks
│  "You are an analyst. Reason from evidence" │  Short, permanent, never changes
├─────────────────────────────────────────────┤
│  Knowledge Files                            │  WHAT the agent knows
│  project.md, schema.md, conventions.md      │  Project context, loaded from files
├─────────────────────────────────────────────┤
│  Skills (Procedures)                        │  WHAT to do for a specific task
│  "1. Pull data 2. Compare 3. Summarize"    │  Step-by-step, activated per task
├─────────────────────────────────────────────┤
│  Working Memory                             │  Runtime scratch space
│  Tool results, findings, artifacts          │  Built up during the session
└─────────────────────────────────────────────┘
```

### Layer 1: System Prompt = Principles

Short. High-level. _Who_ the agent is and _how_ it thinks. This never changes per task.

```python
agent = AgentDefinition(
    name="analyst",
    system_prompt="""You are an analyst.

## Principles
- Reason from evidence, not assumptions.
- Use tools to gather data before drawing conclusions.
- Explain your reasoning alongside every action.
- If results are unclear, dig deeper.
""",
)
```

### Layer 2: Knowledge Files

Project context, domain knowledge, schemas, conventions. Loaded from markdown files on disk. This is the equivalent of a `CLAUDE.md` or similar project knowledge file.

```python
agent = AgentDefinition(
    name="analyst",
    system_prompt="You are an analyst. Reason from evidence.",
    knowledge=[
        "knowledge/project.md",       # project overview, architecture
        "knowledge/schema.md",        # database schema reference
        "knowledge/conventions.md",   # coding conventions, style rules
    ],
)
```

The framework reads these files and injects their contents into the system prompt under a `## Knowledge` section. Files that don't exist are silently skipped.

You can also use glob patterns:

```python
knowledge=["knowledge/*.md"]  # loads all .md files in the knowledge/ dir
```

**Example knowledge file** (`knowledge/project.md`):

```markdown
# Project Overview

This is an e-commerce analytics platform. The main database is PostgreSQL
with tables: orders, customers, products, inventory.

## Key Metrics
- Revenue: sum of order totals
- AOV: average order value
- Churn: customers with no orders in 90 days

## Conventions
- Always filter by tenant_id
- Date ranges are inclusive on both ends
- Currency values are stored in cents
```

### Layer 3: Skills = Procedures

Step-by-step instructions for a specific task. Injected into the prompt when activated.

```python
from corza_agents import Skill, SkillsManager

quarterly_review = Skill(
    name="quarterly-review",
    prompt_template="""
## Quarterly Review Procedure

1. Query revenue data for the current and previous quarter
2. Calculate quarter-over-quarter growth rate
3. Flag any metrics with >20% deviation
4. Produce a summary table with columns: Metric, Q1, Q2, Change%
5. Write a 3-sentence executive summary
""",
    required_tools=["query_revenue", "calculate"],
)

skills = SkillsManager()
skills.register(quarterly_review)
```

Activate skills by name:

```python
agent = AgentDefinition(
    name="analyst",
    system_prompt="You are an analyst. Reason from evidence.",
    knowledge=["knowledge/project.md"],
    skills=["quarterly-review"],
    tools=["query_revenue", "calculate"],
)
```

## Why This Works

**Separation of concerns.** The system prompt is 5 lines. Knowledge lives in files you can edit independently. Procedures live in skills you can version and reuse.

**Reusability.** Same skill across different agents. Same knowledge files across different agents. Mix and match.

**Composability.** Activate multiple skills: `skills=["quarterly-review", "anomaly-detection"]`. Load multiple knowledge files. The framework composes them into one prompt.

**Maintainability.** Edit a knowledge file and every agent that references it picks up the change. No need to update agent definitions.

**Token efficiency.** Only load the knowledge and skills relevant to the current task. Don't dump everything every time.

## Comparison to CLAUDE.md

| CLAUDE.md | Corza Agent Framework |
|-----------|----------------------|
| One monolithic file | Split into principles + knowledge + skills |
| Always fully loaded | Knowledge files loaded per agent, skills activated per task |
| Mixed concerns (identity + context + procedures) | Clean separation |
| Hard to reuse across agents | Knowledge files and skills are reusable |
| Edit the whole file to change one procedure | Edit one skill file |

## Templates

Skill prompts support Jinja2 variables:

```python
skill = Skill(
    name="custom-report",
    prompt_template="Generate a report for {{ company_name }} covering {{ time_period }}.",
    config={"company_name": "Acme", "time_period": "Q1 2025"},
)
```

## Loading Skills

### From Markdown

```python
skill = SkillsManager.from_markdown("review", "Quarterly Review", open("skills/review.md").read())
skills.register(skill)
```

### From Dicts

```python
skills.register(SkillsManager.from_dict({
    "name": "quarterly-review",
    "prompt_template": "...",
    "required_tools": ["query_revenue"],
}))
```

### From Python Functions

The function's docstring becomes the skill prompt:

```python
skill = SkillsManager.from_function(my_analysis_function)
skills.register(skill)
```

### From URLs (async)

Fetch a skill from a remote markdown file at runtime:

```python
skill = await SkillsManager.from_url("https://example.com/skills/review.md")
skills.register(skill)
```

### From Database Queries (async)

Load skills dynamically from your database:

```python
async def load_skill():
    row = await db.fetch_one("SELECT * FROM skills WHERE name = 'review'")
    return {"name": row["name"], "prompt_template": row["template"]}

skill = await SkillsManager.from_database(load_skill)
skills.register(skill)
```

## Runtime Skill Injection

Add or remove skills mid-session through the `ExecutionContext`:

```python
@tool(description="Enable a skill dynamically")
def enable_skill(ctx: ExecutionContext, skill_name: str) -> str:
    new_skill = Skill(name=skill_name, prompt_template="...")
    ctx.add_skill(new_skill)
    return f"Skill '{skill_name}' activated for this session."
```

## Dynamic Knowledge Sources

The `knowledge` field on `AgentDefinition` accepts multiple source types:

```python
agent = AgentDefinition(
    name="analyst",
    model="openai:gpt-5.4",
    knowledge=[
        "knowledge/project.md",                              # file path
        "knowledge/*.md",                                     # glob pattern
        {"title": "DB Schema", "content": "orders(id, ...)"},  # inline dict
        lambda: fetch_schema_from_db(),                        # sync callable
    ],
)
```
