"""`brain-agent eval`: compare agent variants on read-only scenarios.

Each case is a read-only tick with an operator focus; the agent drafts what it
would publish instead of committing. Every variant (a model, optionally with
extra instructions for prompt A/B) runs every case against the live workspace,
starting from a copy of the current notebook so evals never write to the real
one. The workspace moves, so compare variants run in the same eval, not across days.

Scores: cost, Brain calls and failed-program rate from the logs; GroundedIds --
the share of document and job ids in the draft that the agent actually saw in
Brain's outputs; and an LLM judge on groundedness, usefulness/novelty and norms.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

from brain_agent.models import resolve_model
from brain_agent.settings import Settings
from brain_agent.ticks import run_tick

IDS = re.compile(r"\b(d\d{3,}|j[0-9a-f]{32})")
# What the judge reads; GroundedIds always checks against the full transcript.
JUDGE_TRANSCRIPT_CHARS = 150_000

CASES = {
    "choose": "Choose the single most useful contribution you could make right now and draft it in full.",
    "review": "Find a queued or recently submitted sandbox candidate that has no independent review yet "
    "and draft a pre-run review of it.",
    "readout": "Pick the most recently completed full100 sandbox job that has no readout yet and draft one: "
    "score against the current baseline, where it gained or lost, and what it means.",
}

DRAFT_ONLY = (
    "EVALUATION RUN: commits are disabled. Do all the reading and checking you would do, then put the complete "
    "text you would have published -- with its target path -- in the `draft` field of your report."
)


@dataclass
class TickOutput:
    variant: str
    draft: str
    summary: str
    next_step: str
    transcript: str  # possibly cut to its tail for the judge
    seen_ids: list[str]  # every id in the full transcript
    cost_usd: float
    brain_calls: int
    failed_programs: int


def _transcript(log_path: Path) -> tuple[str, int]:
    """Everything the agent saw (start text, programs, outputs) and how many programs failed."""
    if not log_path.exists():
        return "", 0
    parts, failed = [], 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("kind") == "start":
            parts.append(f"[session start]\n{entry.get('output', '')}")
        elif entry.get("kind") == "run":
            failed += entry.get("code") not in (0, None)
            parts.append(f">>> {entry.get('source', '')}\n{entry.get('output', '')}")
        elif entry.get("kind") == "error":
            parts.append(f">>> {entry.get('source', '')}\n[service error] {entry.get('error', '')}")
    return "\n\n".join(parts), failed


@dataclass
class TickMetrics(Evaluator[str, TickOutput, None]):
    def evaluate(self, ctx: EvaluatorContext[str, TickOutput, None]) -> dict[str, float | int]:
        out = ctx.output
        return {
            "cost_usd": round(out.cost_usd, 4),
            "brain_calls": out.brain_calls,
            "failed_rate": round(out.failed_programs / out.brain_calls, 2) if out.brain_calls else 0.0,
            "has_draft": int(bool(out.draft.strip())),
        }


@dataclass
class GroundedIds(Evaluator[str, TickOutput, None]):
    """Share of ids cited in the draft that appear in what Brain actually returned."""

    def evaluate(self, ctx: EvaluatorContext[str, TickOutput, None]) -> float:
        cited = set(IDS.findall(ctx.output.draft))
        if not cited:
            return 1.0
        return round(len(cited & set(ctx.output.seen_ids)) / len(cited), 2)


def judges(model: str) -> list[Evaluator]:
    judge = resolve_model(model)
    return [
        LLMJudge(
            model=judge,
            include_input=True,
            score={"evaluation_name": "grounded", "include_reason": True},
            assertion=False,
            rubric=(
                "The output's `transcript` is everything the agent read from the workspace. Score 0-1 how well "
                "every factual claim and number in `draft` is supported by that transcript. Unsupported or "
                "contradicted claims, invented ids or numbers lower the score sharply."
            ),
        ),
        LLMJudge(
            model=judge,
            include_input=True,
            score={"evaluation_name": "useful", "include_reason": True},
            assertion=False,
            rubric=(
                "Score 0-1 how useful `draft` would be to a community improving one ECOM1 agent together: does it "
                "answer the operator focus (the input), add something not already covered by documents visible in "
                "the transcript, and give concrete, checkable next steps? Duplicating existing work scores low."
            ),
        ),
        LLMJudge(
            model=judge,
            include_input=False,
            score=False,
            assertion={"evaluation_name": "norms", "include_reason": True},
            rubric=(
                "`draft` follows the workspace norms: it names a target path under experiments/, findings/ or "
                "forum/<topic>/, is signed with an agent name and UTC time, cites exact document/job ids, states "
                "what is uncertain, and does not edit or speak for other agents."
            ),
        ),
    ]


def build_dataset(case_names: list[str], judge_model: str) -> Dataset[str, TickOutput, None]:
    return Dataset(
        name="brain-agent read-only scenarios",
        cases=[Case(name=name, inputs=CASES[name]) for name in case_names],
        evaluators=[TickMetrics(), GroundedIds(), *judges(judge_model)],
    )


def eval_task(settings: Settings, variant: str, model: str, extra: str | None, eval_dir: Path):
    async def task(focus: str) -> TickOutput:
        # Each case runs from a copy of the real notebook, in its own state dir.
        state = eval_dir / variant / datetime.now(UTC).strftime("%H%M%S%f")
        state.mkdir(parents=True)
        notebook = settings.state_dir / "memory.sqlite3"
        if notebook.exists():
            shutil.copy2(notebook, state / "memory.sqlite3")
        case_settings = replace(settings, state_dir=state)
        tick = await run_tick(
            case_settings,
            task=f"Eval ({variant}): read-only draft",
            focus=focus,
            max_calls=40,
            model=model,
            extra_instructions="\n\n".join(p for p in (DRAFT_ONLY, extra) if p),
        )
        transcript, failed = _transcript(tick.log_path)
        cost = tick.usage.cost if isinstance(getattr(tick.usage, "cost", None), Decimal) else Decimal(0)
        output = TickOutput(
            variant=variant,
            draft=tick.report.draft or "",
            summary=tick.report.summary,
            next_step=tick.report.next_step,
            transcript=transcript if len(transcript) <= JUDGE_TRANSCRIPT_CHARS
            else "…" + transcript[-JUDGE_TRANSCRIPT_CHARS:],
            seen_ids=sorted(set(IDS.findall(transcript))),
            cost_usd=float(cost),
            brain_calls=tick.deps.calls,
            failed_programs=failed,
        )
        # Keep the draft next to its log so a human can read what was scored.
        (state / "draft.md").write_text(
            f"# {variant}: {focus}\n\n## Summary\n{output.summary}\n\n## Draft\n{output.draft}\n", encoding="utf-8"
        )
        return output

    return task


def parse_variant(spec: str) -> tuple[str, str, str | None]:
    """`name=model` or `name=model@path/to/extra_instructions.md`."""
    name, _, rest = spec.partition("=")
    model, _, extra_path = rest.partition("@")
    extra = Path(extra_path).read_text(encoding="utf-8") if extra_path else None
    return name, model, extra


async def run_evals(
    settings: Settings, variants: list[str], case_names: list[str], judge_model: str, repeat: int = 1
) -> str:
    settings = replace(settings, policy=replace(settings.policy, allow_commit=False, allow_sandbox=False))
    eval_dir = settings.state_dir / "evals" / datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    dataset = build_dataset(case_names, judge_model)
    reports = []
    for spec in variants:
        name, model, extra = parse_variant(spec)
        report = await dataset.evaluate(
            eval_task(settings, name, model, extra, eval_dir), name=name, max_concurrency=1, repeat=repeat
        )
        reports.append(report)
    rendered = []
    for report in reports:
        rendered.append(
            report.render(baseline=reports[0] if report is not reports[0] else None, include_averages=True, width=160)
        )
    text = "\n\n".join(rendered)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "report.txt").write_text(text, encoding="utf-8")
    return text
