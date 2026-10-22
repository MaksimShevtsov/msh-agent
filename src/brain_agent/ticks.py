"""One contribution session (a tick), shared by `tick`, `loop` and `eval`."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import logfire
from pydantic_ai.usage import RunUsage, UsageLimits

from brain_agent.agenda import parse_states, poll_program
from brain_agent.agent import BrainDeps, TickReport, build_agent
from brain_agent.client import DEFAULT_BASE_URL, BrainClient
from brain_agent.settings import Settings

# pydantic-ai stops a run at 50 model requests by default; bigger models take more,
# shorter steps. Spend is bounded by SpendLimits and Brain calls by max_calls.
USAGE_LIMITS = UsageLimits(request_limit=200)


@dataclass
class TickResult:
    stamp: str
    role: str | None
    author: str | None
    report: TickReport
    usage: RunUsage
    deps: BrainDeps
    log_path: Path


class TickStartError(RuntimeError):
    pass


async def run_tick(
    settings: Settings,
    *,
    task: str,
    focus: str | None,
    max_calls: int,
    model: str | None = None,
    advisor_model: str | None = None,
    extra_instructions: str | None = None,
) -> TickResult:
    """Open a Brain session, run the agent once, always leave the session closed.

    `model` / `advisor_model` override the settings for this tick (evals); pass
    advisor_model="" to force no advisor.
    """
    settings = replace(
        settings,
        model=model or settings.model,
        advisor_model=settings.advisor_model if advisor_model is None else (advisor_model or None),
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    agent = build_agent(
        settings.model,
        agent_name=settings.agent_name,
        state_dir=settings.state_dir,
        policy=settings.policy,
        run_budget_usd=settings.run_budget_usd,
        advisor_model=settings.advisor_model,
    )
    async with BrainClient(settings.agent_key, os.environ.get("BRAIN_URL", DEFAULT_BASE_URL)) as client:
        session, start = await client.start(task)
        role, author = start.data.get("role"), start.data.get("author")
        if session is None:
            raise TickStartError(start.output)
        log_path = settings.state_dir / "runs" / f"{stamp}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:  # the start text is part of what the agent saw
            fh.write(json.dumps({"t": time.time(), "kind": "start", "session": session.handle, "task": task,
                                 "output": start.output}, ensure_ascii=False) + "\n")
        deps = settings.deps(client, session=session, log_path=log_path)
        deps.max_calls = max_calls
        prompt = (
            f"UTC now: {stamp}. You are connected as `{author}` (role `{role}`).\n"
            f"Brain call budget for this tick: {max_calls} programs; the next tick comes soon.\n"
            f"Your sandbox jobs so far:\n{deps.ledger.summary()}\n\n"
            f"Your watches (FIRED ones come first on your ladder):\n{settings.agenda.render()}\n\n"
            f"Brain session start text:\n\n{start.output}\n\n"
            f"Operator focus for this tick: {focus or 'your judgement, per the workspace instructions'}"
        )
        try:
            with logfire.span(
                "tick {stamp}", stamp=stamp, session=session.handle, focus=focus,
                model=settings.model, advisor=settings.advisor_model,
            ):
                result = await agent.run(
                    prompt, deps=deps, instructions=extra_instructions, usage_limits=USAGE_LIMITS
                )
        finally:
            if deps.session is not None:
                # Never leave a dangling draft session behind on the shared server.
                await client.run(deps.session, "brain.discard()")
    return TickResult(stamp, role, author, result.output, result.usage, deps, log_path)


async def poll_watches(settings: Settings) -> list[dict]:
    """Between ticks: read the watched jobs' states (no model call); return the watches that fired."""
    jobs = settings.agenda.pending_jobs()
    if not jobs:
        return []
    async with BrainClient(settings.agent_key, os.environ.get("BRAIN_URL", DEFAULT_BASE_URL)) as client:
        session, start = await client.start("Poll watched sandbox jobs (read-only)")
        if session is None:
            raise TickStartError(start.output)
        try:
            result = await client.run(session, poll_program(jobs))
        finally:
            await client.run(session, "brain.discard()")
    return settings.agenda.update_states(parse_states(result.output) if result.ok else {})


def write_report(settings: Settings, tick: TickResult) -> dict:
    report = tick.report.model_dump()
    report.update(
        stamp=tick.stamp, model=settings.model, advisor=settings.advisor_model, role=tick.role,
        calls=tick.deps.calls, usage=str(tick.usage),
    )
    settings.agenda.replace(tick.report.watch, own_jobs=settings.ledger.ids())
    reports = settings.state_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{tick.stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
