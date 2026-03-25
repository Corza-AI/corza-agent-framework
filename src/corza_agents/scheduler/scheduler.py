"""
Corza Agent Framework — Agent Scheduler

DB-backed scheduler for automated agent runs.
Supports cron schedules, one-time runs, and event-driven triggers.

This is a polling-based scheduler — no external dependencies (no Celery, no APScheduler).
Just a background asyncio task that checks the DB for due schedules.

Usage:
    scheduler = AgentScheduler(orchestrator, repository, agent_definitions)

    # Schedule a cron job
    await scheduler.schedule_cron(
        name="daily_report",
        agent_id="analyst",
        cron_expression="0 9 * * *",  # 9am daily
        prompt_template="Generate the daily report for {{ date }}",
    )

    # Schedule a one-time run
    await scheduler.schedule_once(
        name="onboarding_check",
        agent_id="investigator",
        run_at=datetime(2026, 3, 20, 10, 0),
        prompt_template="Check onboarding status for case {{ case_id }}",
        variables={"case_id": "C-12345"},
    )

    # Start the scheduler loop
    await scheduler.start()
"""
import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import and_, select, update

from corza_agents.core.types import AgentDefinition, _now, _uuid
from corza_agents.orchestrator.orchestrator import Orchestrator
from corza_agents.persistence.repository import Repository
from corza_agents.scheduler.models import AgentScheduleModel

log = structlog.get_logger("corza_agents.scheduler")


class ScheduleEntry(BaseModel):
    """Schedule definition."""
    id: str = Field(default_factory=_uuid)
    name: str
    agent_id: str
    schedule_type: str  # "cron", "once", "event"
    cron_expression: str | None = None
    run_at: datetime | None = None
    event_type: str | None = None
    prompt_template: str = ""
    agent_config: dict[str, Any] = Field(default_factory=dict)
    variables: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    max_runs: int | None = None


class AgentScheduler:
    """
    DB-backed agent scheduler.

    Runs as a background task, polling the af_schedules table for due jobs.
    When a schedule is due, it runs the agent and records the result.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        repository: Repository,
        agent_definitions: dict[str, AgentDefinition],
        poll_interval_seconds: int = 30,
    ):
        self._orchestrator = orchestrator
        self._repo = repository
        self._agents = agent_definitions
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None

    # ══════════════════════════════════════════════════════════════════
    # Schedule Management
    # ══════════════════════════════════════════════════════════════════

    async def schedule_cron(
        self,
        name: str,
        agent_id: str,
        cron_expression: str,
        prompt_template: str,
        variables: dict | None = None,
        metadata: dict | None = None,
        max_runs: int | None = None,
    ) -> str:
        """Schedule a recurring cron-based agent run."""
        entry = ScheduleEntry(
            name=name,
            agent_id=agent_id,
            schedule_type="cron",
            cron_expression=cron_expression,
            prompt_template=prompt_template,
            variables=variables or {},
            metadata=metadata or {},
            max_runs=max_runs,
        )
        next_run = self._next_cron_run(cron_expression)
        return await self._save_schedule(entry, next_run)

    async def schedule_once(
        self,
        name: str,
        agent_id: str,
        run_at: datetime,
        prompt_template: str,
        variables: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Schedule a one-time agent run at a specific time."""
        entry = ScheduleEntry(
            name=name,
            agent_id=agent_id,
            schedule_type="once",
            run_at=run_at,
            prompt_template=prompt_template,
            variables=variables or {},
            metadata=metadata or {},
            max_runs=1,
        )
        return await self._save_schedule(entry, run_at)

    async def schedule_on_event(
        self,
        name: str,
        agent_id: str,
        event_type: str,
        prompt_template: str,
        variables: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Schedule an agent run triggered by an event."""
        entry = ScheduleEntry(
            name=name,
            agent_id=agent_id,
            schedule_type="event",
            event_type=event_type,
            prompt_template=prompt_template,
            variables=variables or {},
            metadata=metadata or {},
        )
        return await self._save_schedule(entry, next_run_at=None)

    async def trigger_event(self, event_type: str, event_data: dict | None = None) -> list[str]:
        """
        Trigger all event-based schedules matching the event type.
        Returns list of session IDs for the triggered runs.
        """
        async with self._repo.session() as db:
            result = await db.execute(
                select(AgentScheduleModel).where(
                    and_(
                        AgentScheduleModel.enabled == True,  # noqa
                        AgentScheduleModel.schedule_type == "event",
                        AgentScheduleModel.event_type == event_type,
                    )
                )
            )
            schedules = result.scalars().all()

        session_ids = []
        for schedule in schedules:
            variables = dict(schedule.variables or {})
            if event_data:
                variables.update(event_data)
            sid = await self._execute_schedule(schedule, variables)
            if sid:
                session_ids.append(sid)
        return session_ids

    async def cancel_schedule(self, schedule_id: str) -> None:
        """Disable a schedule."""
        async with self._repo.session() as db:
            await db.execute(
                update(AgentScheduleModel)
                .where(AgentScheduleModel.id == schedule_id)
                .values(enabled=False, updated_at=_now())
            )
            await db.commit()

    async def list_schedules(self, enabled_only: bool = True) -> list[dict]:
        """List all schedules."""
        async with self._repo.session() as db:
            query = select(AgentScheduleModel)
            if enabled_only:
                query = query.where(AgentScheduleModel.enabled == True)  # noqa
            result = await db.execute(query.order_by(AgentScheduleModel.created_at))
            return [
                {
                    "id": s.id, "name": s.name, "agent_id": s.agent_id,
                    "type": s.schedule_type, "cron": s.cron_expression,
                    "event_type": s.event_type, "enabled": s.enabled,
                    "next_run_at": s.next_run_at.isoformat() if s.next_run_at else None,
                    "last_run_at": s.last_run_at.isoformat() if s.last_run_at else None,
                    "last_run_status": s.last_run_status,
                    "run_count": s.run_count,
                }
                for s in result.scalars().all()
            ]

    # ══════════════════════════════════════════════════════════════════
    # Scheduler Loop
    # ══════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        log.info("scheduler_started", poll_interval=self._poll_interval)

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("scheduler_stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop — checks for due schedules."""
        while self._running:
            try:
                await self._check_due_schedules()
            except Exception as e:
                log.error("scheduler_poll_error", error=str(e)[:500])
            await asyncio.sleep(self._poll_interval)

    async def _check_due_schedules(self) -> None:
        """Find and execute all schedules that are due."""
        now = datetime.now(UTC)
        async with self._repo.session() as db:
            result = await db.execute(
                select(AgentScheduleModel).where(
                    and_(
                        AgentScheduleModel.enabled == True,  # noqa
                        AgentScheduleModel.schedule_type.in_(["cron", "once"]),
                        AgentScheduleModel.next_run_at <= now,
                    )
                )
            )
            due_schedules = result.scalars().all()

        for schedule in due_schedules:
            log.info("schedule_due", id=schedule.id, name=schedule.name, type=schedule.schedule_type)
            asyncio.create_task(self._execute_and_update(schedule))

    async def _execute_and_update(self, schedule: AgentScheduleModel) -> None:
        """Execute a schedule and update its state in DB."""
        session_id = await self._execute_schedule(schedule)

        async with self._repo.session() as db:
            values: dict[str, Any] = {
                "last_run_at": _now(),
                "last_run_session_id": session_id,
                "last_run_status": "completed" if session_id else "failed",
                "run_count": schedule.run_count + 1,
                "updated_at": _now(),
            }

            # Calculate next run
            if schedule.schedule_type == "cron" and schedule.cron_expression:
                if schedule.max_runs and schedule.run_count + 1 >= schedule.max_runs:
                    values["enabled"] = False
                else:
                    values["next_run_at"] = self._next_cron_run(schedule.cron_expression)
            elif schedule.schedule_type == "once":
                values["enabled"] = False

            await db.execute(
                update(AgentScheduleModel)
                .where(AgentScheduleModel.id == schedule.id)
                .values(**values)
            )
            await db.commit()

    async def _execute_schedule(
        self,
        schedule: AgentScheduleModel,
        extra_variables: dict | None = None,
    ) -> str | None:
        """Execute a scheduled agent run."""
        agent_def = self._agents.get(schedule.agent_id)
        if not agent_def:
            log.error("schedule_agent_not_found", agent_id=schedule.agent_id, schedule_id=schedule.id)
            return None

        variables = dict(schedule.variables or {})
        if extra_variables:
            variables.update(extra_variables)
        variables["_schedule_id"] = schedule.id
        variables["_schedule_name"] = schedule.name
        variables["_run_time"] = datetime.now(UTC).isoformat()

        # Render prompt template
        from corza_agents.prompts.templates import render_template
        prompt = render_template(schedule.prompt_template, variables)

        session_id = _uuid()
        metadata = dict(schedule.metadata_ if hasattr(schedule, "metadata_") else (schedule.metadata or {}))
        metadata["scheduled"] = True
        metadata["schedule_id"] = schedule.id

        try:
            async for event in self._orchestrator.run(
                session_id=session_id,
                user_message=prompt,
                agent_def=agent_def,
                metadata=metadata,
                variables=variables,
            ):
                if event.type.value == "error":
                    log.warning("scheduled_run_error",
                               schedule_id=schedule.id,
                               session_id=session_id,
                               error=event.data.get("message"))

            log.info("scheduled_run_completed",
                     schedule_id=schedule.id,
                     session_id=session_id)
            return session_id

        except Exception as e:
            log.error("scheduled_run_failed",
                      schedule_id=schedule.id,
                      error=str(e)[:500])
            return None

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    async def _save_schedule(self, entry: ScheduleEntry, next_run_at: datetime | None) -> str:
        """Persist a schedule entry to DB."""
        async with self._repo.session() as db:
            model = AgentScheduleModel(
                id=entry.id,
                name=entry.name,
                agent_id=entry.agent_id,
                schedule_type=entry.schedule_type,
                cron_expression=entry.cron_expression,
                run_at=entry.run_at,
                event_type=entry.event_type,
                prompt_template=entry.prompt_template,
                agent_config=entry.agent_config,
                variables=entry.variables,
                metadata_=entry.metadata,
                enabled=entry.enabled,
                next_run_at=next_run_at,
                max_runs=entry.max_runs,
            )
            db.add(model)
            await db.commit()

        log.info("schedule_created",
                 id=entry.id, name=entry.name,
                 type=entry.schedule_type,
                 next_run=next_run_at.isoformat() if next_run_at else None)
        return entry.id

    @staticmethod
    def _next_cron_run(cron_expression: str) -> datetime:
        """
        Calculate the next run time for a cron expression.
        Uses croniter if available, falls back to simple interval parsing.
        """
        try:
            from croniter import croniter
            cron = croniter(cron_expression, datetime.now(UTC))
            return cron.get_next(datetime).replace(tzinfo=UTC)
        except ImportError:
            # Fallback: parse simple patterns
            parts = cron_expression.strip().split()
            if len(parts) >= 5:
                minute = int(parts[0]) if parts[0] != "*" else 0
                hour = int(parts[1]) if parts[1] != "*" else 0
                now = datetime.now(UTC)
                next_run = now.replace(minute=minute, second=0, microsecond=0)
                if parts[1] != "*":
                    next_run = next_run.replace(hour=hour)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run
            return datetime.now(UTC) + timedelta(hours=1)
