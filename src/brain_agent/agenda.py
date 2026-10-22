"""What the agent is waiting on, kept across ticks in state/agenda.json.

Each tick ends by declaring its watches (TickReport.watch). Between ticks the
loop polls the watched sandbox jobs with one cheap read-only program -- no model
call -- and starts the next tick early when one of them finishes. The tick then
sees the fired watches at the top of its prompt and acts on them first.

The agent may drop its own watches, but never one on its own unfinished job.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

JOB_ID = re.compile(r"^j[0-9a-f]{8,}$")
ACTIVE = {"queued", "running"}
UNKNOWN = "unknown"


class Watch(BaseModel):
    ref: str = Field(
        description="A sandbox job id (j...) to be woken on when it finishes, or a document id (d...) to re-check."
    )
    action: str = Field(description="What you will do when it fires, e.g. 'readout: t062 deltas vs jd8713c77'.")


def poll_program(job_ids: list[str]) -> str:
    """One Brain program that prints {job: state} for the given jobs."""
    return (
        "import json\n"
        "out = {}\n"
        f"for j in {job_ids!r}:\n"
        "    try:\n"
        "        out[j] = brain.sandbox.status(j).get('state')\n"
        "    except Exception:\n"
        f"        out[j] = {UNKNOWN!r}\n"
        "print(json.dumps(out))"
    )


def parse_states(output: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", output, re.DOTALL)
    return {k: str(v) for k, v in json.loads(match[0]).items()} if match else {}


class Agenda:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []

    def _save(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def replace(self, watches: list[Watch], own_jobs: set[str]) -> None:
        """Take the tick's declared watches; keep what is known about refs already watched."""
        old = {item["ref"]: item for item in self.load()}
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        items = [
            {**old.get(w.ref, {"added": stamp, "state": None, "fired": False}), "ref": w.ref, "action": w.action}
            for w in watches
        ]
        declared = {w.ref for w in watches}
        for job in sorted(own_jobs - declared):
            item = old.get(job)
            if item is None:
                items.append({"ref": job, "action": "readout of your own sandbox job", "added": stamp,
                              "state": None, "fired": False})
            elif not item["fired"]:
                items.append(item)
        self._save(items)

    def pending_jobs(self) -> list[str]:
        return [i["ref"] for i in self.load() if JOB_ID.match(i["ref"]) and not i["fired"]]

    def update_states(self, states: dict[str, str]) -> list[dict]:
        """Record polled states; return the watches that fired now (their job left the queue)."""
        items, fired = self.load(), []
        for item in items:
            state = states.get(item["ref"])
            if state is None:
                continue
            item["state"] = state
            if not item["fired"] and state not in ACTIVE and state != UNKNOWN:
                item["fired"] = True
                fired.append(item)
        self._save(items)
        return fired

    def render(self) -> str:
        items = self.load()
        if not items:
            return "(nothing watched)"
        fired = [i for i in items if i["fired"]]
        rest = [i for i in items if not i["fired"]]
        lines = [f"- FIRED {i['ref']} (state {i['state']}): {i['action']}" for i in fired]
        lines += [f"- {i['ref']} (last seen {i['state'] or 'not polled yet'}): {i['action']}" for i in rest]
        return "\n".join(lines)
