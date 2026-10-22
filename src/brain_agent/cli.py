"""`brain-agent tick | loop | serve`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime

import logfire
from dotenv import load_dotenv

from brain_agent.agent import build_agent
from brain_agent.client import DEFAULT_BASE_URL, BrainClient
from brain_agent.observability import configure as configure_observability
from brain_agent.scorecard import scorecard
from brain_agent.settings import ROOT, Settings


async def tick(args: argparse.Namespace) -> int:
    settings = Settings.from_env(read_only=args.read_only)
    if not settings.agent_key:
        print("BRAIN_AGENT_KEY is not set (get one at https://bitgn.com/me/agents)", file=sys.stderr)
        return 2
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    agent = build_agent(
        settings.model,
        agent_name=settings.agent_name,
        state_dir=settings.state_dir,
        policy=settings.policy,
        run_budget_usd=settings.run_budget_usd,
    )
    async with BrainClient(settings.agent_key, os.environ.get("BRAIN_URL", DEFAULT_BASE_URL)) as client:
        session, start = await client.start(args.task)
        role = start.data.get("role")
        print(f"session {session and session.handle} role={role} author={start.data.get('author')}")
        if session is None:
            print(start.output, file=sys.stderr)
            return 1
        deps = settings.deps(client, session=session, log_path=settings.state_dir / "runs" / f"{stamp}.jsonl")
        deps.max_calls = args.max_calls
        prompt = (
            f"UTC now: {stamp}. You are connected as `{start.data.get('author')}` (role `{role}`).\n"
            f"Your sandbox jobs so far:\n{deps.ledger.summary()}\n\n"
            f"Brain session start text:\n\n{start.output}\n\n"
            f"Operator focus for this tick: {args.focus or 'your judgement, per the workspace instructions'}"
        )
        try:
            with logfire.span("tick {stamp}", stamp=stamp, session=session.handle, focus=args.focus):
                result = await agent.run(prompt, deps=deps)
        finally:
            if deps.session is not None:
                # Never leave a dangling draft session behind on the shared server.
                await client.run(deps.session, "brain.discard()")
    report = result.output.model_dump()
    report.update(stamp=stamp, model=settings.model, role=role, calls=deps.calls, usage=str(result.usage))
    (settings.state_dir / "reports").mkdir(parents=True, exist_ok=True)
    (settings.state_dir / "reports" / f"{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


async def run_scorecard(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.offline or not settings.agent_key:
        text = await scorecard(settings.state_dir, None)
    else:
        async with BrainClient(settings.agent_key, os.environ.get("BRAIN_URL", DEFAULT_BASE_URL)) as client:
            text = await scorecard(settings.state_dir, client)
    (settings.state_dir / "scorecard.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


async def loop(args: argparse.Namespace) -> int:
    """Tick forever, `--hours` apart; one failed tick never stops the loop."""
    while True:
        try:
            await tick(args)
        except Exception as exc:  # logged, then wait for the next slot
            print(f"tick failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.stdout.flush()
        await asyncio.sleep(args.hours * 3600)


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(prog="brain-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("tick", "run one contribution session"), ("loop", "tick every --hours")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--task", default="Contribute to the shared ECOM1 research")
        p.add_argument("--focus", default=None, help="operator hint for this tick")
        p.add_argument("--read-only", action="store_true", help="force commits and sandbox off")
        p.add_argument("--max-calls", type=int, default=60)
        if name == "loop":
            p.add_argument("--hours", type=float, default=3.0)
    p = sub.add_parser("scorecard", help="summarize ticks and check published work in Brain")
    p.add_argument("--offline", action="store_true", help="local logs only, no Brain session")
    p = sub.add_parser("serve", help="AG-UI server for the CopilotKit chat")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    configure_observability(f"msh-agent-{'chat' if args.command == 'serve' else 'loop'}")
    if args.command == "serve":
        from brain_agent.server import serve

        serve(args.host, args.port)
        return
    if args.command == "scorecard":
        sys.exit(asyncio.run(run_scorecard(args)))
    sys.exit(asyncio.run(loop(args) if args.command == "loop" else tick(args)))
