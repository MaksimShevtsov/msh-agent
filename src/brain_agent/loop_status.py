"""Read-only view of the scheduled loop, for the chat agent.

Reads only what the loop writes under the state dir -- the latest report, the
newest tick log and the job ledger -- and returns a bounded text summary. It is
the chat's single window onto local files; the agent gets no general file access.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from brain_agent.sandbox_guard import JobLedger

MAX_SOURCE_CHARS = 160


def _stamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def _newest(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime) if directory.is_dir() else []
    return files[-1] if files else None


def _entry_line(entry: dict) -> str:
    when = datetime.fromtimestamp(entry.get("t", 0), UTC).strftime("%H:%M:%SZ")
    kind = entry.get("kind", "?")
    first = (entry.get("source") or entry.get("task") or "").strip().splitlines()
    source = (first[0] if first else "")[:MAX_SOURCE_CHARS]
    detail = entry.get("error") or entry.get("reason") or ""
    code = f" code={entry['code']}" if "code" in entry else ""
    return f"  {when} {kind}{code}: {source}" + (f"  -- {str(detail)[:200]}" if detail else "")


def loop_status(state_dir: Path, tail: int = 12) -> str:
    report_path = _newest(state_dir / "reports", "*.json")
    log_path = _newest(state_dir / "runs", "*.jsonl")
    lines: list[str] = [f"Now: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ')}"]

    in_progress = log_path is not None and (
        report_path is None or log_path.stat().st_mtime > report_path.stat().st_mtime
    )
    if log_path is not None:
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        runs = sum(1 for e in entries if e.get("kind") == "run")
        errors = sum(1 for e in entries if e.get("kind") == "error")
        state = "IN PROGRESS" if in_progress else "finished"
        lines.append(
            f"\nNewest tick log: {log_path.name} ({state}; last write {_stamp(log_path)}; "
            f"{runs} programs, {errors} service errors)"
        )
        lines.append(f"Last {min(tail, len(entries))} entries:")
        lines.extend(_entry_line(e) for e in entries[-tail:])
    else:
        lines.append("\nNo tick has run yet.")

    if report_path is not None:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        lines.append(f"\nLatest finished tick report: {report_path.name}")
        for key in ("summary", "contribution", "committed", "next_step", "calls"):
            if key in report:
                lines.append(f"  {key}: {report[key]}")

    lines.append(f"\nSandbox jobs in the ledger:\n{JobLedger(state_dir / 'jobs.json').summary()}")
    return "\n".join(lines)
