"""Tests for all persistence backends."""
import pytest

from corza_agents import (
    AgentMessage,
    AgentSession,
    MessageRole,
    SessionStatus,
    create_repository,
)
from corza_agents.persistence.base import BaseRepository
from corza_agents.persistence.memory import InMemoryRepository


@pytest.fixture
def memory_repo():
    return InMemoryRepository()


@pytest.fixture
def sqlite_repo(tmp_path):
    from corza_agents.persistence.sqlite import SQLiteRepository
    return SQLiteRepository(db_path=str(tmp_path / "test.db"))


# ── Factory ───────────────────────────────────────────────────────


def test_factory_memory():
    repo = create_repository("memory")
    assert isinstance(repo, BaseRepository)
    assert isinstance(repo, InMemoryRepository)


def test_factory_sqlite(tmp_path):
    repo = create_repository("sqlite", db_path=str(tmp_path / "test.db"))
    assert isinstance(repo, BaseRepository)


def test_factory_invalid():
    with pytest.raises(ValueError, match="Unknown backend"):
        create_repository("redis")


# ── Shared backend tests ─────────────────────────────────────────

async def _test_session_crud(repo: BaseRepository):
    await repo.initialize()
    session = AgentSession(id="s1", agent_id="agent_a", user_id="u1", tenant_id="t1")
    await repo.create_session(session)

    s = await repo.get_session("s1")
    assert s is not None
    assert s.agent_id == "agent_a"
    assert s.user_id == "u1"
    assert s.tenant_id == "t1"
    assert s.status == SessionStatus.IDLE

    await repo.update_session("s1", status=SessionStatus.RUNNING)
    s = await repo.get_session("s1")
    assert s.status == SessionStatus.RUNNING

    assert await repo.get_session("nonexistent") is None
    await repo.close()


async def _test_user_sessions(repo: BaseRepository):
    await repo.initialize()
    await repo.create_session(AgentSession(id="s1", agent_id="a", user_id="alice", tenant_id="acme"))
    await repo.create_session(AgentSession(id="s2", agent_id="a", user_id="alice", tenant_id="acme"))
    await repo.create_session(AgentSession(id="s3", agent_id="a", user_id="bob", tenant_id="acme"))

    alice_sessions = await repo.get_sessions_for_user("alice", "acme")
    assert len(alice_sessions) == 2

    bob_sessions = await repo.get_sessions_for_user("bob", "acme")
    assert len(bob_sessions) == 1

    all_acme = await repo.get_sessions_for_user("alice")
    assert len(all_acme) == 2

    empty = await repo.get_sessions_for_user("nobody")
    assert len(empty) == 0
    await repo.close()


async def _test_delete_session(repo: BaseRepository):
    await repo.initialize()
    await repo.create_session(AgentSession(id="s1", agent_id="a"))
    msg = AgentMessage(id="m1", session_id="s1", role=MessageRole.USER, content="Hello")
    await repo.add_message(msg)

    assert await repo.get_session("s1") is not None
    await repo.delete_session("s1")
    assert await repo.get_session("s1") is None
    await repo.close()


async def _test_messages(repo: BaseRepository):
    await repo.initialize()
    await repo.create_session(AgentSession(id="s1", agent_id="a"))

    msg1 = AgentMessage(id="m1", session_id="s1", role=MessageRole.USER, content="Hello")
    msg2 = AgentMessage(id="m2", session_id="s1", role=MessageRole.ASSISTANT, content="Hi")
    await repo.add_message(msg1)
    await repo.add_message(msg2)

    msgs = await repo.get_messages("s1")
    assert len(msgs) == 2
    assert msgs[0].content == "Hello"
    assert msgs[1].content == "Hi"

    await repo.mark_messages_summarized("s1", ["m1"])
    msgs = await repo.get_messages("s1", include_summarized=False)
    assert len(msgs) == 1
    assert msgs[0].id == "m2"
    await repo.close()


async def _test_memory(repo: BaseRepository):
    await repo.initialize()
    await repo.set_memory("agent1", "key1", {"data": "value"})
    val = await repo.get_memory("agent1", "key1")
    assert val == {"data": "value"}

    await repo.set_memory("agent1", "key1", "updated")
    val = await repo.get_memory("agent1", "key1")
    assert val == "updated"

    mems = await repo.list_memories("agent1")
    assert len(mems) == 1

    assert await repo.get_memory("agent1", "missing") is None
    await repo.close()


async def _test_artifacts(repo: BaseRepository):
    await repo.initialize()
    await repo.create_session(AgentSession(id="s1", agent_id="a"))

    aid = await repo.save_artifact("s1", "report", "my_report", content="Report text")
    assert aid

    arts = await repo.get_artifacts("s1")
    assert len(arts) == 1
    assert arts[0]["name"] == "my_report"

    arts = await repo.get_artifacts("s1", artifact_type="other")
    assert len(arts) == 0
    await repo.close()


async def _test_audit(repo: BaseRepository):
    await repo.initialize()
    await repo.create_session(AgentSession(id="s1", agent_id="a"))

    await repo.log_audit("s1", "test_event", "test_action", actor="tester")
    log = await repo.get_audit_log("s1")
    assert len(log) == 1
    assert log[0]["event_type"] == "test_event"
    await repo.close()


async def _test_tool_execution(repo: BaseRepository):
    await repo.initialize()
    await repo.create_session(AgentSession(id="s1", agent_id="a"))

    await repo.log_tool_execution(
        session_id="s1", message_id=None, tool_call_id="tc1",
        tool_name="search", input_data={"q": "test"},
        output_data={"results": []}, status="success",
        duration_ms=42.0,
    )
    await repo.close()


# ── InMemoryRepository ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_sessions(memory_repo):
    await _test_session_crud(memory_repo)

@pytest.mark.asyncio
async def test_memory_user_sessions(memory_repo):
    await _test_user_sessions(memory_repo)

@pytest.mark.asyncio
async def test_memory_delete_session(memory_repo):
    await _test_delete_session(memory_repo)

@pytest.mark.asyncio
async def test_memory_messages(memory_repo):
    await _test_messages(memory_repo)

@pytest.mark.asyncio
async def test_memory_memory(memory_repo):
    await _test_memory(memory_repo)

@pytest.mark.asyncio
async def test_memory_artifacts(memory_repo):
    await _test_artifacts(memory_repo)

@pytest.mark.asyncio
async def test_memory_audit(memory_repo):
    await _test_audit(memory_repo)

@pytest.mark.asyncio
async def test_memory_tool_execution(memory_repo):
    await _test_tool_execution(memory_repo)


# ── SQLiteRepository ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sqlite_sessions(sqlite_repo):
    await _test_session_crud(sqlite_repo)

@pytest.mark.asyncio
async def test_sqlite_user_sessions(sqlite_repo):
    await _test_user_sessions(sqlite_repo)

@pytest.mark.asyncio
async def test_sqlite_delete_session(sqlite_repo):
    await _test_delete_session(sqlite_repo)

@pytest.mark.asyncio
async def test_sqlite_messages(sqlite_repo):
    await _test_messages(sqlite_repo)

@pytest.mark.asyncio
async def test_sqlite_memory(sqlite_repo):
    await _test_memory(sqlite_repo)

@pytest.mark.asyncio
async def test_sqlite_artifacts(sqlite_repo):
    await _test_artifacts(sqlite_repo)

@pytest.mark.asyncio
async def test_sqlite_audit(sqlite_repo):
    await _test_audit(sqlite_repo)

@pytest.mark.asyncio
async def test_sqlite_tool_execution(sqlite_repo):
    await _test_tool_execution(sqlite_repo)
