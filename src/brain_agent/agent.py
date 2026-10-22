"""The Brain agent: Pydantic AI + harness capabilities around one remote-code tool.

Brain already *is* code mode -- every call is a Python program executed by Monty
on the server -- so the agent needs no local CodeMode sandbox. The harness
supplies what a long research session lacks: a plan, a notebook that survives
between scheduled ticks, a spend brake, and context management.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai_harness.advisor import Advisor
from pydantic_ai_harness.compaction import SummarizingCompaction
from pydantic_ai_harness.memory import Memory, SqliteMemoryStore
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments
from pydantic_ai_harness.spend import Budget, SpendLimits
from pydantic_ai_harness.tool_output_limits import LocalFileStore, ToolOutputLimits

from brain_agent.agenda import Watch
from brain_agent.approvals import APPROVAL_TOOL, Approvals
from brain_agent.claims import Claims
from brain_agent.client import BrainClient, BrainError, RunResult, Session
from brain_agent.loop_status import loop_status as read_loop_status
from brain_agent.models import resolve_model
from brain_agent.policy import Policy, check_source, closes_session
from brain_agent.prompts import render_instructions
from brain_agent.sandbox_guard import (
    JobLedger,
    SandboxRules,
    check_cancel,
    check_queue,
    plan_submit,
    submitted_jobs,
)


@dataclass
class BrainDeps:
    client: BrainClient
    policy: Policy
    session: Session | None
    log_path: Path
    ledger: JobLedger
    rules: SandboxRules = field(default_factory=SandboxRules)
    calls: int = 0
    max_calls: int = 60
    committed: list[str] = field(default_factory=list)
    # Submissions staged in the open session; they reach the queue (and the
    # ledger) only when that session commits.
    staged_jobs: list[dict] = field(default_factory=list)
    submits: int = 0
    last_run_submitted: bool = False
    # Interactive (chat) runs: commit and submit wait for the operator's click.
    approvals: Approvals | None = None
    # Shared with this agent's other process (loop <-> chat) so both never take the same work.
    claims: Claims | None = None


class TickReport(BaseModel):
    """What this session did, for the operator's log."""

    summary: str = Field(description="Two to five sentences: what you looked at and what you did.")
    contribution: str | None = Field(
        default=None, description="Path(s) of documents you created or edited, or null if none."
    )
    committed: bool = Field(description="True only if brain.commit() succeeded in this session.")
    next_step: str = Field(description="The next useful thing for whoever (or whichever tick) comes next.")
    draft: str | None = Field(
        default=None,
        description="Read-only runs only: the full text you would have published (with its target path), else null.",
    )
    watch: list[Watch] = Field(
        default_factory=list,
        description="Everything you are waiting on. Job ids are polled every few minutes and wake you when they "
        "finish; this list replaces the previous one, so repeat what you still watch.",
    )


def format_result(result: RunResult) -> str:
    parts = [result.output.rstrip() or "(no output)"]
    for notice in result.notices:
        parts.append(f"\n[workspace notice -- read before writing]\n{notice.get('content', notice)}")
    if result.truncated:
        parts.append(
            f"\n[output truncated: page it with brain.output({result.request!r}, "
            f"offset={result.data.get('next_offset')})]"
        )
    if not result.ok:
        parts.append(f"\n[run failed with code {result.code}; its writes were rolled back]")
    return "\n".join(parts)


def _log(deps: BrainDeps, kind: str, **fields: object) -> None:
    deps.log_path.parent.mkdir(parents=True, exist_ok=True)
    with deps.log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.time(), "kind": kind, **fields}, ensure_ascii=False) + "\n")


async def _diff(deps: BrainDeps) -> str:
    assert deps.session is not None
    try:
        result = await deps.client.run(deps.session, "print(brain.diff())")
    except BrainError as exc:
        return f"(could not read the diff: {exc})"
    return result.output


async def _queue_depth(deps: BrainDeps) -> int | None:
    assert deps.session is not None
    try:
        result = await deps.client.run(deps.session, "print(len(brain.sandbox.queue()))")
    except BrainError:
        return None
    _log(deps, "queue", session=deps.session.handle, code=result.code, output=result.output)
    match = re.match(r"\s*(\d+)", result.output)
    return int(match[1]) if result.ok and match else None


def build_agent(
    model: str | Model,
    *,
    agent_name: str,
    state_dir: Path,
    policy: Policy,
    run_budget_usd: Decimal,
    with_loop_status: bool = False,
    advisor_model: str | None = None,
) -> Agent[BrainDeps, TickReport]:
    # Hybrid: a cheap executor does the reading; a stronger advisor weighs in at the two
    # decisions that matter. Local mode with a resolved Model keeps zai on its coding
    # endpoint; no history is forwarded, so each consultation carries its own evidence.
    advisor = (
        [Advisor(resolve_model(advisor_model), mode="local", max_uses=1, max_tokens=4096)] if advisor_model else []
    )
    agent = Agent(
        resolve_model(model) if isinstance(model, str) else model,
        deps_type=BrainDeps,
        output_type=TickReport,
        instructions=render_instructions(
            agent_name, policy.allow_commit, policy.allow_sandbox, with_advisor=bool(advisor_model)
        ),
        capabilities=[
            *advisor,
            # Stores that live on this host directly: workspace-backed ones need a
            # LocalWorkspace, which is POSIX-only, and the agent needs no host tools anyway.
            Planning(tools=["write_plan", "read_plan", "update_task_status"]),
            Memory(SqliteMemoryStore(database=state_dir / "memory.sqlite3"), agent_name=agent_name),
            SpendLimits(budgets=[Budget(usd=run_budget_usd, window="run")]),
            ToolOutputLimits(store=LocalFileStore(base_dir=state_dir / "overflow")),
            SummarizingCompaction(max_fraction=0.6, keep_messages=16),
            # Small models (glm-flash) emit malformed tool JSON now and then.
            RepairToolArguments(),
        ],
        defer_model_check=True,
    )

    @agent.tool
    async def brain_run(ctx: RunContext[BrainDeps], source: str) -> str:
        """Execute a Python program on the Brain server in your task session.

        `brain` is pre-bound; use print() to see results. Keep each program small
        (4 s wall clock). Returns printed output plus any workspace notices.
        """
        deps = ctx.deps
        if deps.session is None:
            return "No open session (the last one was committed or discarded). Call brain_start first."
        if deps.calls >= deps.max_calls:
            return "Call budget for this tick is spent. Finish: discard or commit, then write your report."
        handle = deps.session.handle

        def refuse(reason: str) -> str:
            _log(deps, "refused", session=handle, source=source, reason=reason)
            return f"[refused by operator policy] {reason}"

        refusal = check_source(source, deps.policy) or check_cancel(source, deps.ledger.ids())
        if refusal:
            return refuse(refusal)
        plan = plan_submit(source)
        if isinstance(plan, str):
            return refuse(plan)
        sent = source
        if plan is not None:
            if deps.submits >= deps.rules.max_submits_per_tick:
                return refuse(f"At most {deps.rules.max_submits_per_tick} sandbox submission(s) per tick.")
            depth = await _queue_depth(deps)
            if depth is None:
                return refuse("Could not read the sandbox queue depth; not submitting blind.")
            if queue_refusal := check_queue(plan, depth, deps.rules):
                return refuse(queue_refusal)
            sent = plan.instrumented(source)
        if deps.approvals is not None and (plan is not None or re.search(r"\bcommit\s*\(", source)):
            action = "sandbox submit" if plan is not None else "commit"
            decision = deps.approvals.decision(ctx.messages, handle, source)
            if decision is None:
                diff = await _diff(deps) if action == "commit" else ""
                approval_id = deps.approvals.request(handle, source, action, diff)
                _log(deps, "approval_requested", session=handle, source=source, approval_id=approval_id)
                return (
                    f"[approval required] This {action} waits for the operator. Call the "
                    f"`{APPROVAL_TOOL}` tool with approval_id={approval_id!r}, then, if approved, "
                    "send this exact program again unchanged."
                )
            if decision is False:
                return refuse(f"The operator rejected this {action}.")
        deps.calls += 1
        try:
            result = await deps.client.run(deps.session, sent)
        except BrainError as exc:
            _log(deps, "error", session=handle, source=sent, error=str(exc))
            return f"[service error] {exc}"
        _log(deps, "run", session=handle, source=sent, code=result.code, output=result.output)
        if result.ok and re.search(r"\bundo\s*\(", source) and deps.last_run_submitted:
            deps.staged_jobs.pop()
            deps.submits -= 1
        deps.last_run_submitted = False
        if result.ok and plan is not None:
            for job in submitted_jobs(result.output):
                deps.staged_jobs.append({"job": job, "trials": plan.trials, "session": handle})
                deps.submits += 1
                deps.last_run_submitted = True
        if result.ok and closes_session(source):
            if re.search(r"\bcommit\s*\(", source):
                deps.committed.append(handle)
                deps.ledger.add(deps.staged_jobs)
            else:  # discarded: staged submissions never reached the queue
                deps.submits -= len(deps.staged_jobs)
            deps.staged_jobs = []
            deps.session = None
        return format_result(result)

    @agent.tool
    async def brain_start(ctx: RunContext[BrainDeps], task: str) -> str:
        """Open a new Brain task session (needed only after a commit or discard closed the last one)."""
        deps = ctx.deps
        if deps.session is not None:
            return f"Session {deps.session.handle} is still open; finish it (commit or discard) first."
        session, result = await deps.client.start(task)
        deps.session = session
        _log(deps, "start", session=session and session.handle, task=task, role=result.data.get("role"))
        return format_result(result)

    @agent.tool
    async def claim_work(ctx: RunContext[BrainDeps], refs: list[str], what: str) -> str:
        """Claim documents/jobs (ids like d3066, j...) before working on them, so your other
        process (the scheduled loop or the operator chat) does not do the same work. Refused
        when the other process holds a live claim on any of them: then pick something else."""
        if ctx.deps.claims is None:
            return "No claims store in this run; go ahead."
        return ctx.deps.claims.claim(refs, what)

    if with_loop_status:

        @agent.tool_plain
        def loop_status() -> str:
            """Status of the scheduled loop on this machine: whether a tick is running now, its latest
            programs and errors, the last finished tick's report, your sandbox job ledger and watches.
            Read-only; local state only (use brain_run for the workspace itself)."""
            return read_loop_status(state_dir)

    return agent
