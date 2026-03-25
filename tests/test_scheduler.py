"""
Tests for AgentScheduler — schedule creation, listing, and model validation.

Uses an in-memory SQLite database so no external PostgreSQL is needed.
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from corza_agents.core.types import AgentDefinition
from corza_agents.persistence.models import Base
from corza_agents.scheduler.models import AgentScheduleModel
from corza_agents.scheduler.scheduler import AgentScheduler, ScheduleEntry

# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def db_engine():
    """Create an in-memory SQLite async engine with all tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def mock_repo(db_engine):
    """
    A mock Repository whose .session() context manager yields real
    SQLAlchemy async sessions backed by in-memory SQLite.
    """
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    class _FakeRepo:
        def session(self):
            return session_factory()

    return _FakeRepo()


@pytest_asyncio.fixture
async def mock_orchestrator():
    """A minimal mock Orchestrator — scheduler only calls .run() on execution."""
    orch = MagicMock()
    orch.run = AsyncMock(return_value=iter([]))
    return orch


@pytest_asyncio.fixture
async def agent_defs():
    """Sample agent definitions for the scheduler."""
    return {
        "analyst": AgentDefinition(
            name="analyst",
            model="mock:test",
            description="Analyzes data",
        ),
        "researcher": AgentDefinition(
            name="researcher",
            model="mock:test",
            description="Researches topics",
        ),
    }


@pytest_asyncio.fixture
async def scheduler(mock_orchestrator, mock_repo, agent_defs):
    return AgentScheduler(
        orchestrator=mock_orchestrator,
        repository=mock_repo,
        agent_definitions=agent_defs,
        poll_interval_seconds=60,
    )


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════


def test_scheduler_creation(mock_orchestrator, mock_repo, agent_defs):
    """AgentScheduler can be instantiated without error."""
    sched = AgentScheduler(
        orchestrator=mock_orchestrator,
        repository=mock_repo,
        agent_definitions=agent_defs,
    )
    assert sched is not None
    assert sched._running is False
    assert sched._task is None


def test_schedule_model_creation():
    """AgentScheduleModel can be instantiated with expected fields."""
    model = AgentScheduleModel(
        id="sched-001",
        name="daily_report",
        agent_id="analyst",
        schedule_type="cron",
        cron_expression="0 9 * * *",
        prompt_template="Generate daily report for {{ date }}",
        enabled=True,
    )
    assert model.id == "sched-001"
    assert model.name == "daily_report"
    assert model.agent_id == "analyst"
    assert model.schedule_type == "cron"
    assert model.cron_expression == "0 9 * * *"
    assert model.enabled is True
    # run_count default is applied at DB flush time; on raw instantiation it's None
    assert model.run_count is None or model.run_count == 0


def test_schedule_entry_pydantic():
    """ScheduleEntry (Pydantic model) validates and defaults correctly."""
    entry = ScheduleEntry(
        name="weekly_check",
        agent_id="researcher",
        schedule_type="cron",
        cron_expression="0 8 * * 1",
        prompt_template="Weekly status check",
    )
    assert entry.name == "weekly_check"
    assert entry.enabled is True
    assert entry.variables == {}
    assert entry.metadata == {}
    assert entry.max_runs is None
    assert entry.id  # auto-generated UUID


@pytest.mark.asyncio
async def test_scheduler_schedule_cron(scheduler, mock_repo, db_engine):
    """schedule_cron persists a cron schedule and returns an ID."""
    schedule_id = await scheduler.schedule_cron(
        name="daily_report",
        agent_id="analyst",
        cron_expression="0 9 * * *",
        prompt_template="Generate the daily report",
    )

    assert schedule_id is not None
    assert isinstance(schedule_id, str)
    assert len(schedule_id) > 0

    # Verify it was persisted
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(AgentScheduleModel).where(AgentScheduleModel.id == schedule_id)
        )
        row = result.scalar_one_or_none()
        assert row is not None
        assert row.name == "daily_report"
        assert row.schedule_type == "cron"
        assert row.cron_expression == "0 9 * * *"


@pytest.mark.asyncio
async def test_scheduler_schedule_once(scheduler, mock_repo, db_engine):
    """schedule_once persists a one-time schedule at the given datetime."""
    future_time = datetime.now(UTC) + timedelta(hours=2)
    schedule_id = await scheduler.schedule_once(
        name="onboarding_check",
        agent_id="researcher",
        run_at=future_time,
        prompt_template="Check onboarding for {{ case_id }}",
        variables={"case_id": "C-99"},
    )

    assert schedule_id is not None

    # Verify it was persisted
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(AgentScheduleModel).where(AgentScheduleModel.id == schedule_id)
        )
        row = result.scalar_one_or_none()
        assert row is not None
        assert row.name == "onboarding_check"
        assert row.schedule_type == "once"
        assert row.max_runs == 1
        assert row.variables.get("case_id") == "C-99"


@pytest.mark.asyncio
async def test_scheduler_list_schedules(scheduler, mock_repo, db_engine):
    """list_schedules returns all enabled schedules."""
    await scheduler.schedule_cron(
        name="schedule_a",
        agent_id="analyst",
        cron_expression="0 9 * * *",
        prompt_template="Report A",
    )
    await scheduler.schedule_cron(
        name="schedule_b",
        agent_id="researcher",
        cron_expression="0 17 * * *",
        prompt_template="Report B",
    )

    schedules = await scheduler.list_schedules()

    assert len(schedules) == 2
    names = {s["name"] for s in schedules}
    assert "schedule_a" in names
    assert "schedule_b" in names
