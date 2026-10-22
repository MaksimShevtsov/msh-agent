"""`brain-agent scorecard`: how the agent is doing, from its logs and from Brain.

Local part (state/reports + state/runs): per tick cost, Brain calls, failed
programs, service errors (timeouts), policy refusals, whether it committed.

Brain part (one read-only session, discarded at the end):
* citations -- other workspace documents that mention each document the agent
  published (the community actually using its work);
* watched jobs -- sandbox jobs its reports point at, with their current state and
  score, so its predictions about them can be checked once they finish.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from brain_agent.client import BrainClient

JOB_ID = re.compile(r"\bj[0-9a-f]{32}\b")
# Documents the agent authored live under these trees; roles/ and LANES.md are shared ledgers it only appends to.
OWN_PATH = re.compile(r"\b((?:experiments|forum|findings)/[\w./\-]+?\.(?:md|py|json))\b")
CREATE_CALL = re.compile(r"brain\.create\(\s*[\"']w1[\"']\s*,\s*[\"']([^\"']+)[\"']")
COST = re.compile(r"cost=Decimal\('([0-9.]+)'\)")


@dataclass
class TickStats:
    stamp: str
    committed: bool
    cost: Decimal
    calls: int
    failed_programs: int
    service_errors: int
    timeouts: int
    refusals: int
    minutes: float
    jobs: list[str] = field(default_factory=list)
    steers: int = 0


def _log_stats(path: Path) -> tuple[int, int, int, int, float, int]:
    if not path.exists():
        return 0, 0, 0, 0, 0.0, 0
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = sum(1 for e in entries if e.get("kind") == "run" and e.get("code") not in (0, None))
    errors = [e for e in entries if e.get("kind") == "error"]
    timeouts = sum(1 for e in errors if "wall-clock" in e.get("error", ""))
    refusals = sum(1 for e in entries if e.get("kind") == "refused")
    steers = sum(1 for e in entries if e.get("kind") == "review" and e.get("verdict") == "steer")
    times = [e["t"] for e in entries if "t" in e]
    minutes = (max(times) - min(times)) / 60 if len(times) > 1 else 0.0
    return failed, len(errors), timeouts, refusals, minutes, steers


def collect_ticks(state_dir: Path) -> list[TickStats]:
    ticks = []
    for report_path in sorted((state_dir / "reports").glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stamp = report.get("stamp", report_path.stem)
        failed, errors, timeouts, refusals, minutes, steers = _log_stats(state_dir / "runs" / f"{stamp}.jsonl")
        cost = COST.search(report.get("usage", ""))
        ticks.append(
            TickStats(
                stamp=stamp,
                committed=bool(report.get("committed")),
                cost=Decimal(cost[1]) if cost else Decimal(0),
                calls=int(report.get("calls", 0)),
                failed_programs=failed,
                service_errors=errors,
                timeouts=timeouts,
                refusals=refusals,
                minutes=minutes,
                jobs=sorted(set(JOB_ID.findall(json.dumps(report)))),
                steers=steers,
            )
        )
    return ticks


def published_paths(state_dir: Path) -> list[str]:
    """Workspace paths the agent created in sessions that committed (ticks and chat)."""
    paths: set[str] = set()
    for report_path in (state_dir / "reports").glob("*.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("committed"):
            paths.update(OWN_PATH.findall(report.get("contribution") or ""))
    for log in [*(state_dir / "runs").glob("*.jsonl"), *(state_dir / "chat").glob("*.jsonl")]:
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        committed = {
            e.get("session")
            for e in entries
            if e.get("kind") == "run" and e.get("code") == 0 and re.search(r"\bcommit\s*\(", e.get("source", ""))
        }
        for e in entries:
            if e.get("kind") == "run" and e.get("code") == 0 and e.get("session") in committed:
                paths.update(p for p in CREATE_CALL.findall(e.get("source", "")) if OWN_PATH.fullmatch(p))
    return sorted(paths)


def local_section(ticks: list[TickStats]) -> list[str]:
    lines = [
        "## Ticks",
        "",
        (
            "| tick | committed | cost $ | Brain calls | failed programs | service errors (timeouts) | refusals "
            "| steers | minutes |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in ticks:
        lines.append(
            f"| {t.stamp} | {'yes' if t.committed else 'no'} | {t.cost:.3f} | {t.calls} | {t.failed_programs} "
            f"| {t.service_errors} ({t.timeouts}) | {t.refusals} | {t.steers} | {t.minutes:.0f} |"
        )
    if ticks:
        total = sum((t.cost for t in ticks), Decimal(0))
        committed = sum(t.committed for t in ticks)
        calls = sum(t.calls for t in ticks) or 1
        lines += [
            "",
            f"**{len(ticks)} ticks**, {committed} committed ({committed / len(ticks):.0%}); "
            f"total ${total:.2f}, ${total / len(ticks):.3f}/tick; "
            f"failed programs {sum(t.failed_programs for t in ticks) / calls:.0%} of calls; "
            f"timeouts {sum(t.timeouts for t in ticks)}; refusals {sum(t.refusals for t in ticks)}; "
            f"judge steers {sum(t.steers for t in ticks)}.",
        ]
    return lines


# Runs inside Brain (Monty): one SDK call per program keeps each under the 4 s limit.
CITATIONS_PROGRAM = """
import json
page = brain.search_matches("w1", {pattern!r}, regex=True, limit=50)
print(json.dumps([m["path"] for m in page["matches"] if m["id"] != {doc!r}]))
"""
STAT_PROGRAM = 'import json\nprint(json.dumps(brain.stat({ref!r})))'
JOB_PROGRAM = """
import json
s = brain.sandbox.status({job!r})
out = {{"name": s.get("name"), "author": s.get("author"), "state": s.get("state")}}
if s.get("result_available"):
    r = brain.sandbox.result({job!r})["result"]
    out.update(score=r.get("run_score"), completed=r.get("completed_trials"), selected=r.get("selected_trials"))
print(json.dumps(out))
"""


async def brain_section(client: BrainClient, state_dir: Path, ticks: list[TickStats]) -> list[str]:
    session, _ = await client.start("Scorecard: read-only check of msh-agent's published work")
    if session is None:
        return ["## Brain", "", "(could not open a Brain session)"]
    lines = [
        "## Citations of published documents",
        "",
        "External citations only: the agent's own documents and roles/review.md (where it appends its entries) are excluded.",
        "",
    ]
    own = published_paths(state_dir)
    self_citing = set(own) | {"roles/review.md"}
    try:
        for path in own:
            stat = await client.run(session, STAT_PROGRAM.format(ref=f"w1:{path}"))
            if not stat.ok:
                lines.append(f"- {path}: not found in the workspace (moved, deleted or never committed)")
                continue
            doc = json.loads(stat.output)["id"]
            found = await client.run(session, CITATIONS_PROGRAM.format(pattern=rf"\b{doc}\b|{re.escape(path)}", doc=doc))
            citing = [p for p in json.loads(found.output) if p not in self_citing] if found.ok else None
            if citing is None:
                lines.append(f"- `{doc}` {path}: search failed ({found.output.strip()[:120]})")
            else:
                shown = ", ".join(citing[:5]) + (" …" if len(citing) > 5 else "")
                lines.append(f"- `{doc}` {path}: **{len(citing)}** other documents cite it" + (f" ({shown})" if citing else ""))
        lines += ["", "## Watched sandbox jobs (referenced in reports)", ""]
        for job in sorted({j for t in ticks for j in t.jobs}):
            status = await client.run(session, JOB_PROGRAM.format(job=job))
            lines.append(f"- `{job[:12]}…` {status.output.strip() if status.ok else 'status unavailable'}")
    finally:
        await client.run(session, "brain.discard()")
    return lines


async def scorecard(state_dir: Path, client: BrainClient | None) -> str:
    ticks = collect_ticks(state_dir)
    lines = ["# msh-agent scorecard", ""] + local_section(ticks)
    if client is not None:
        lines += [""] + await brain_section(client, state_dir, ticks)
    return "\n".join(lines) + "\n"
