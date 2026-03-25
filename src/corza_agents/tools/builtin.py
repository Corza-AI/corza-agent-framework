"""
Corza Agent Framework — Built-in Tools

The agent's self-management capabilities. These tools are how agents
plan, delegate, record findings, and maintain persistent knowledge.

Architecture: Always Orchestrator + Sub-Agents.
- The orchestrator plans and delegates via `manage_agent`
- Both orchestrator and sub-agents manage their plan via `manage_plan`
- Session-scoped scratch notes via `manage_notes`
- Permanently persisted documents via `manage_knowledge` (survives across sessions)
- Persistent objective via `manage_objective` (the agent's mission / soul.md)
- Context compaction via `manage_context`

Built-in tools:
- manage_objective:  The agent's mission and purpose (persistent, auto-loaded)
- manage_agent:      Delegate work to / communicate with sub-agents
- manage_plan:       Structured todo list (session-scoped)
- manage_notes:      Session scratch pad (lost when session ends)
- manage_knowledge:  Persistent documents (knowledge, findings — survives across sessions)
- manage_skill:      Reusable procedures (persistent, self-optimizing)
- manage_context:    Manually trigger context compaction
"""
import json

from corza_agents.core.types import ExecutionContext
from corza_agents.tools.decorators import tool

# ══════════════════════════════════════════════════════════════════════
# Objective — The Agent's Mission (persistent, auto-loaded into prompt)
# ══════════════════════════════════════════════════════════════════════


@tool(
    name="manage_objective",
    description=(
        "Read or update your objective — your mission, purpose, and constraints. "
        "This is your north star: WHY you exist and WHAT you're optimizing for. "
        "The objective is automatically loaded into your system prompt every turn. "
        "Actions: 'read' (view current), 'write' (replace entirely), 'append' (add to end). "
        "Update it when the mission evolves. It persists permanently across sessions."
    ),
    tags=["objective", "persistent", "self-management"],
)
async def manage_objective(
    action: str,
    content: str = "",
    ctx: ExecutionContext = None,
) -> dict:
    """
    Manage the agent's persistent objective (soul.md / program.md / CLAUDE.md).

    The objective is a special document that is ALWAYS auto-loaded into the
    system prompt — full content, every turn. Unlike knowledge documents
    (which show only names in the prompt and must be read on demand), the
    objective is always present in the agent's context.

    It defines:
    - The agent's mission and purpose
    - What it's optimizing for
    - Domain constraints and boundaries
    - Key definitions and terminology
    - Success criteria

    The objective persists permanently across all sessions for this agent.

    Args:
        action: One of 'read', 'write', 'append'
        content: The objective content (for write/append)
    """
    if not ctx or not ctx.repository:
        return {"status": "error", "message": "No repository available"}

    agent_id = ctx.agent_id
    key = "objective"

    if action == "read":
        value = await ctx.repository.get_memory(agent_id, key)
        if value is None:
            return {
                "status": "success", "content": None,
                "message": "No objective set. Use manage_objective(action='write') to define your mission.",
            }
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return {"status": "success", "content": text, "size": len(text)}

    if action == "write":
        if not content:
            return {"status": "error", "message": "Provide 'content' for the objective."}
        await ctx.repository.set_memory(
            agent_id, key, content,
            memory_type="objective", session_id=ctx.session_id,
        )
        return {"status": "success", "action": "written", "size": len(content)}

    if action == "append":
        if not content:
            return {"status": "error", "message": "Provide 'content' to append."}
        existing = await ctx.repository.get_memory(agent_id, key)
        existing_str = existing if isinstance(existing, str) else ""
        separator = "\n\n" if existing_str else ""
        updated = existing_str + separator + content
        await ctx.repository.set_memory(
            agent_id, key, updated,
            memory_type="objective", session_id=ctx.session_id,
        )
        return {"status": "success", "action": "appended", "size": len(updated)}

    return {"status": "error", "message": f"Unknown action '{action}'. Use: read, write, append."}


# ══════════════════════════════════════════════════════════════════════
# Delegation
# ══════════════════════════════════════════════════════════════════════

@tool(
    name="manage_agent",
    description=(
        "Full lifecycle management of sub-agents. Two-way communication. "
        "Actions: "
        "'spawn' — dispatch ONE sub-agent (blocks until completion, returns result); "
        "'spawn_parallel' — dispatch MULTIPLE sub-agents concurrently (pass JSON array of "
        "{agent_name, task} objects in 'tasks' param — runs all in parallel, returns all results); "
        "'message' — send a follow-up to an existing sub-agent (refine, redirect, ask more); "
        "'report' — (used BY sub-agents) send interim findings back to the orchestrator; "
        "'status' — check a sub-agent's current state; "
        "'list' — show all available sub-agent types. "
        "Sub-agent sessions persist — the orchestrator can return to them across turns."
    ),
    timeout_seconds=300,
    tags=["orchestration", "delegation"],
)
async def manage_agent(
    action: str,
    agent_name: str = "",
    task: str = "",
    tasks: str = "",
    session_id: str = "",
    context_data: str = "",
    content: str = "",
    ctx: ExecutionContext = None,
) -> dict:
    """
    Manage sub-agent lifecycle with two-way communication.

    Args:
        action: One of 'spawn', 'spawn_parallel', 'message', 'report', 'status', 'list'
        agent_name: Name of the sub-agent to spawn (for 'spawn')
        task: Task description (for 'spawn') or follow-up message (for 'message')
        tasks: JSON array of {agent_name, task, context_data?} objects (for 'spawn_parallel')
        session_id: Child session ID (for 'message' and 'status')
        context_data: Optional JSON context to pass (for 'spawn')
        content: Report content (for 'report' — used by sub-agents to send findings back)

    This is a placeholder — the actual implementation is wired by the
    Orchestrator at runtime with access to SubAgentRunner and definitions.
    """
    return {
        "status": "error",
        "message": "manage_agent requires orchestrator wiring. "
                   "Use Orchestrator.run() instead of AgentEngine.run() directly.",
    }


# ══════════════════════════════════════════════════════════════════════
# Plan Management (session-scoped)
# ══════════════════════════════════════════════════════════════════════

@tool(
    name="manage_plan",
    description=(
        "Manage your execution plan (todo list). "
        "Actions: 'add' (new item), 'update' (change status/description), "
        "'complete' (mark done), 'remove' (delete item), 'clear' (reset entire plan), 'list' (show all). "
        "Your plan is your discipline mechanism. Update it after every action. "
        "It should always reflect what's done, what's in progress, and what remains."
    ),
    tags=["planning", "self-management"],
)
async def manage_plan(
    action: str,
    item: str = "",
    status: str = "",
    item_id: str = "",
    ctx: ExecutionContext = None,
) -> dict:
    """
    Manage a structured plan in working memory (session-scoped).

    Args:
        action: One of 'add', 'update', 'complete', 'remove', 'clear', 'list'
        item: Description of the plan item (for add/update)
        status: Status: 'pending', 'in_progress', 'done', 'blocked' (for update)
        item_id: ID of existing item (for update/complete/remove). Use '1', '2', etc.
    """
    if not ctx or not ctx.working_memory:
        return {"status": "error", "message": "No working memory available"}

    wm = ctx.working_memory
    plan = wm.get("_plan") or []

    if action == "list":
        if not plan:
            return {"status": "success", "plan": [], "message": "Plan is empty."}
        return {"status": "success", "plan": plan, "count": len(plan)}

    if action == "add":
        if not item:
            return {"status": "error", "message": "Provide 'item' description to add."}
        entry = {"id": str(len(plan) + 1), "item": item, "status": "pending"}
        plan.append(entry)
        wm.store("_plan", plan)
        return {"status": "success", "action": "added", "entry": entry, "total": len(plan)}

    if action == "update":
        idx = _resolve_index(item_id, plan)
        if idx is None:
            return {"status": "error", "message": f"Item '{item_id}' not found."}
        if item:
            plan[idx]["item"] = item
        if status:
            plan[idx]["status"] = status
        wm.store("_plan", plan)
        return {"status": "success", "action": "updated", "entry": plan[idx]}

    if action == "complete":
        idx = _resolve_index(item_id, plan)
        if idx is None:
            return {"status": "error", "message": f"Item '{item_id}' not found."}
        plan[idx]["status"] = "done"
        wm.store("_plan", plan)
        remaining = [p for p in plan if p["status"] != "done"]
        return {
            "status": "success", "action": "completed", "entry": plan[idx],
            "remaining": len(remaining), "total": len(plan),
        }

    if action == "remove":
        idx = _resolve_index(item_id, plan)
        if idx is None:
            return {"status": "error", "message": f"Item '{item_id}' not found."}
        removed = plan.pop(idx)
        for i, p in enumerate(plan):
            p["id"] = str(i + 1)
        wm.store("_plan", plan)
        return {"status": "success", "action": "removed", "removed": removed}

    if action == "clear":
        count = len(plan)
        wm.store("_plan", [])
        return {"status": "success", "action": "cleared", "items_removed": count}

    return {"status": "error", "message": f"Unknown action '{action}'. Use: add, update, complete, remove, clear, list."}


def _resolve_index(item_id: str, plan: list) -> int | None:
    """Find a plan item by ID or index."""
    if not item_id:
        return None
    for i, p in enumerate(plan):
        if p.get("id") == item_id:
            return i
    try:
        idx = int(item_id) - 1
        if 0 <= idx < len(plan):
            return idx
    except ValueError:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════
# Notes (session-scoped scratch pad)
# ══════════════════════════════════════════════════════════════════════

@tool(
    name="manage_notes",
    description=(
        "Manage session scratch notes. These are temporary — lost when the session ends. "
        "Actions: 'read', 'write' (replace), 'append', 'clear'. "
        "Use for in-progress reasoning, intermediate calculations, or scratch work. "
        "For permanent knowledge, use manage_knowledge instead."
    ),
    tags=["notes", "session", "self-management"],
)
async def manage_notes(
    action: str,
    content: str = "",
    ctx: ExecutionContext = None,
) -> dict:
    """
    Session-scoped scratch notes in working memory.

    Args:
        action: One of 'read', 'write', 'append', 'clear'
        content: Text to write or append
    """
    if not ctx or not ctx.working_memory:
        return {"status": "error", "message": "No working memory available"}

    wm = ctx.working_memory
    key = "_notes"

    if action == "read":
        notes = wm.get(key) or ""
        return {"status": "success", "notes": notes, "size": len(notes)}

    if action == "write":
        wm.store(key, content)
        return {"status": "success", "action": "written", "size": len(content)}

    if action == "append":
        existing = wm.get(key) or ""
        separator = "\n" if existing else ""
        updated = existing + separator + content
        wm.store(key, updated)
        return {"status": "success", "action": "appended", "size": len(updated)}

    if action == "clear":
        wm.store(key, "")
        return {"status": "success", "action": "cleared"}

    return {"status": "error", "message": f"Unknown action '{action}'. Use: read, write, append, clear."}


# ══════════════════════════════════════════════════════════════════════
# Documents (permanently persisted across sessions)
# ══════════════════════════════════════════════════════════════════════

@tool(
    name="manage_knowledge",
    description=(
        "Manage your knowledge library — a folder of named .md documents that "
        "persist permanently across sessions. Each document is a separate file. "
        "Actions: 'read' (load a document), 'write' (create or replace), "
        "'append' (add to end), 'list' (show all documents), 'delete' (remove). "
        "Use 'findings' for investigation results, 'knowledge' for accumulated facts, "
        "or create any named document you need."
    ),
    tags=["knowledge", "persistent", "self-management"],
)
async def manage_knowledge(
    action: str,
    name: str = "",
    content: str = "",
    ctx: ExecutionContext = None,
) -> dict:
    """
    Manage a folder of persistent knowledge documents.

    Each agent has its own knowledge folder in the database:
      agent/
        └── knowledge/
              ├── findings.md
              ├── schema-notes.md
              ├── customer-segments.md
              └── (any name you create)

    Documents persist across all sessions for this agent.

    Args:
        action: One of 'read', 'write', 'append', 'list', 'delete'
        name: Document name (e.g., 'findings', 'knowledge', 'schema-notes')
        content: Content to write or append
    """
    if not ctx or not ctx.repository:
        return {"status": "error", "message": "No repository available"}

    agent_id = ctx.agent_id
    doc_prefix = "doc:"

    if action == "list":
        all_memories = await ctx.repository.list_memories(agent_id, memory_type="document")
        doc_list = [
            {"name": m["key"].removeprefix(doc_prefix), "size": len(str(m.get("value", "")))}
            for m in all_memories
            if m.get("key", "").startswith(doc_prefix)
        ]
        return {"status": "success", "documents": doc_list, "count": len(doc_list)}

    if action == "read":
        if not name:
            return {"status": "error", "message": "Provide 'name' of document to read."}
        value = await ctx.repository.get_memory(agent_id, doc_prefix + name)
        if value is None:
            return {"status": "success", "name": name, "content": None,
                    "message": f"Document '{name}' does not exist yet."}
        doc_content = value if isinstance(value, str) else json.dumps(value, default=str)
        return {"status": "success", "name": name, "content": doc_content, "size": len(doc_content)}

    if action == "write":
        if not name:
            return {"status": "error", "message": "Provide 'name' of document to write."}
        await ctx.repository.set_memory(
            agent_id, doc_prefix + name, content,
            memory_type="document", session_id=ctx.session_id,
        )
        return {"status": "success", "action": "written", "name": name, "size": len(content)}

    if action == "append":
        if not name:
            return {"status": "error", "message": "Provide 'name' of document to append to."}
        existing = await ctx.repository.get_memory(agent_id, doc_prefix + name)
        existing_str = existing if isinstance(existing, str) else ""
        separator = "\n" if existing_str else ""
        updated = existing_str + separator + content
        await ctx.repository.set_memory(
            agent_id, doc_prefix + name, updated,
            memory_type="document", session_id=ctx.session_id,
        )
        return {"status": "success", "action": "appended", "name": name, "size": len(updated)}

    if action == "delete":
        if not name:
            return {"status": "error", "message": "Provide 'name' of document to delete."}
        await ctx.repository.set_memory(
            agent_id, doc_prefix + name, None,
            memory_type="document", session_id=ctx.session_id,
        )
        return {"status": "success", "action": "deleted", "name": name}

    return {"status": "error", "message": f"Unknown action '{action}'. Use: read, write, append, list, delete."}


# ══════════════════════════════════════════════════════════════════════
# Skill Management (permanently persisted, self-optimizing)
# ══════════════════════════════════════════════════════════════════════

@tool(
    name="manage_skill",
    description=(
        "Manage reusable skills (proven procedures) that persist across sessions. "
        "Actions: 'read' (load full skill content), 'write' (create/update a skill), "
        "'list' (show all skills), 'delete' (remove a skill). "
        "Skills are step-by-step procedures for complex tasks. "
        "When a workflow succeeds, save it as a skill. Next time, load and follow it. "
        "Skills stored permanently in the database, scoped to the agent."
    ),
    tags=["skills", "persistent", "self-optimization"],
)
async def manage_skill(
    action: str,
    name: str = "",
    description: str = "",
    content: str = "",
    required_tools: str = "",
    ctx: ExecutionContext = None,
) -> dict:
    """
    Manage persistent skills stored in the database.
    Scoped to agent_id — available across all sessions.

    Args:
        action: One of 'read', 'write', 'list', 'delete'
        name: Skill name (e.g., 'quarterly-review', 'data-validation')
        description: 1-line description of what the skill does (for 'write')
        content: Full skill content — step-by-step procedure (for 'write')
        required_tools: Comma-separated list of tool names needed (for 'write')
    """
    if not ctx or not ctx.repository:
        return {"status": "error", "message": "No repository available"}

    agent_id = ctx.agent_id
    skill_prefix = "skill:"

    if action == "list":
        all_memories = await ctx.repository.list_memories(agent_id, memory_type="skill")
        skill_list = []
        for m in all_memories:
            if m.get("key", "").startswith(skill_prefix):
                val = m.get("value", {})
                if isinstance(val, dict):
                    skill_list.append({
                        "name": m["key"].removeprefix(skill_prefix),
                        "description": val.get("description", ""),
                        "required_tools": val.get("required_tools", []),
                    })
        return {"status": "success", "skills": skill_list, "count": len(skill_list)}

    if action == "read":
        if not name:
            return {"status": "error", "message": "Provide 'name' of skill to read."}
        value = await ctx.repository.get_memory(agent_id, skill_prefix + name)
        if value is None:
            return {"status": "success", "name": name, "content": None,
                    "message": f"Skill '{name}' does not exist yet."}
        if isinstance(value, dict):
            return {
                "status": "success", "name": name,
                "description": value.get("description", ""),
                "content": value.get("content", ""),
                "required_tools": value.get("required_tools", []),
            }
        return {"status": "success", "name": name, "content": str(value)}

    if action == "write":
        if not name:
            return {"status": "error", "message": "Provide 'name' for the skill."}
        if not content:
            return {"status": "error", "message": "Provide 'content' (the procedure) for the skill."}
        tools_list = [t.strip() for t in required_tools.split(",") if t.strip()] if required_tools else []
        skill_data = {
            "description": description or name,
            "content": content,
            "required_tools": tools_list,
        }
        await ctx.repository.set_memory(
            agent_id, skill_prefix + name, skill_data,
            memory_type="skill", session_id=ctx.session_id,
        )
        return {"status": "success", "action": "written", "name": name,
                "description": description or name}

    if action == "delete":
        if not name:
            return {"status": "error", "message": "Provide 'name' of skill to delete."}
        await ctx.repository.set_memory(
            agent_id, skill_prefix + name, None,
            memory_type="skill", session_id=ctx.session_id,
        )
        return {"status": "success", "action": "deleted", "name": name}

    return {"status": "error", "message": f"Unknown action '{action}'. Use: read, write, list, delete."}


# ══════════════════════════════════════════════════════════════════════
# Context Management
# ══════════════════════════════════════════════════════════════════════

@tool(
    name="manage_context",
    description=(
        "Manually trigger context compaction to free up context window space. "
        "Use when running low on context and need to summarize older history. "
        "Compaction runs at the start of the NEXT turn — older messages are "
        "summarized by the LLM and replaced with a concise summary."
    ),
    tags=["memory", "context"],
)
async def manage_context(ctx: ExecutionContext = None) -> dict:
    """Trigger manual context compaction on the next turn."""
    if not ctx or not ctx.working_memory:
        return {"status": "error", "message": "No working memory available"}

    # Set the flag that the engine checks at the top of each turn
    ctx.working_memory.store("_compact_requested", True)
    return {
        "status": "success",
        "message": "Context compaction scheduled. Older messages will be summarized on the next turn.",
    }


# ══════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════

BUILTIN_TOOLS = [
    manage_objective,
    manage_agent,
    manage_plan,
    manage_notes,
    manage_knowledge,
    manage_skill,
    manage_context,
]
