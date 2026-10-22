"""Rules for the shared ECOM1 sandbox queue, enforced in code.

Community norms (forum/coordination, 2026-10-22): above ~10 queued jobs only
smoke runs, above ~20 new full100s hold. On top of those, this agent:

* submits at most once per tick, in the one recognisable form
  ``job = brain.sandbox.submit(...)`` with a literal ``trials=`` -- so the job id
  can be captured and the size checked before anything is sent;
* sends a full run (more than ``smoke_max`` trials) only when the live queue is
  at most ``full_depth_max`` deep, and nothing at all above ``hard_depth_max``;
* cancels only jobs recorded in its own ledger, by literal id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SUBMIT_CALL = re.compile(r"\bsubmit\s*\(")
SUBMIT_LINE = re.compile(r"^(?P<var>[A-Za-z_]\w*)\s*=\s*brain\.sandbox\.submit\((?P<args>.*)\)\s*$", re.M)
TRIALS_ARG = re.compile(r"\btrials\s*=\s*(?P<n>\d+)\b")
CANCEL_CALL = re.compile(r"\bcancel\s*\(")
CANCEL_LITERAL = re.compile(r"\bbrain\.sandbox\.cancel\(\s*[\"'](?P<job>j[0-9a-f]+)[\"']")
JOB_MARKER = "SUBMITTED_JOB"
JOB_LINE = re.compile(rf"{JOB_MARKER}\s+(?P<job>j[0-9a-f]+)")


@dataclass(frozen=True)
class SandboxRules:
    max_submits_per_tick: int = 1
    smoke_max: int = 10
    full_depth_max: int = 10
    hard_depth_max: int = 25


@dataclass(frozen=True)
class SubmitPlan:
    var: str
    trials: int

    def instrumented(self, source: str) -> str:
        # Print the id so the client can record it whatever the model prints.
        return f"{source}\nprint({JOB_MARKER!r}, {self.var})"


def plan_submit(source: str) -> SubmitPlan | str | None:
    """None: no submission here. str: refusal. SubmitPlan: allowed shape."""
    calls = SUBMIT_CALL.findall(source)
    if not calls:
        return None
    lines = list(SUBMIT_LINE.finditer(source))
    if len(calls) != 1 or len(lines) != 1:
        return (
            "Submit exactly once per program, on its own line, as "
            "`job = brain.sandbox.submit(\"dID\", revision=REV, trials=N, name=\"...\")`."
        )
    trials = TRIALS_ARG.search(lines[0]["args"])
    if not trials:
        return "Pass a literal `trials=N` so the queue rules can be checked."
    return SubmitPlan(var=lines[0]["var"], trials=int(trials["n"]))


def check_queue(plan: SubmitPlan, depth: int, rules: SandboxRules) -> str | None:
    if depth > rules.hard_depth_max:
        return f"The shared queue is {depth} deep (limit {rules.hard_depth_max}); do not submit now."
    if plan.trials > rules.smoke_max and depth > rules.full_depth_max:
        return (
            f"The shared queue is {depth} deep: only smoke runs (trials <= {rules.smoke_max}) "
            f"until it is at most {rules.full_depth_max}. Submit a smoke run or wait for a later tick."
        )
    return None


def check_cancel(source: str, own_jobs: set[str]) -> str | None:
    calls = len(CANCEL_CALL.findall(source))
    if not calls:
        return None
    literals = [m["job"] for m in CANCEL_LITERAL.finditer(source)]
    if len(literals) != calls:
        return 'Cancel only as `brain.sandbox.cancel("jID", reason="...")` with a literal job id.'
    foreign = [job for job in literals if job not in own_jobs]
    if foreign:
        return f"Not this agent's jobs (see your job ledger): {', '.join(foreign)}."
    return None


def submitted_jobs(output: str) -> list[str]:
    return [m["job"] for m in JOB_LINE.finditer(output)]


class JobLedger:
    """Jobs this agent has committed, kept across ticks in state/jobs.json."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []

    def ids(self) -> set[str]:
        return {entry["job"] for entry in self.load()}

    def add(self, entries: list[dict]) -> None:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        jobs = self.load() + [{**entry, "committed_at": stamp} for entry in entries]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    def summary(self, limit: int = 10) -> str:
        jobs = self.load()[-limit:]
        if not jobs:
            return "(no sandbox jobs submitted yet)"
        return "\n".join(
            f"- {j['job']} trials={j['trials']} committed {j['committed_at']} (session {j['session']})" for j in jobs
        )
