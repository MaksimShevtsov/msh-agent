"""`brain-agent tick | loop | scorecard | eval | serve`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

from brain_agent.client import DEFAULT_BASE_URL, BrainClient
from brain_agent.observability import configure as configure_observability
from brain_agent.scorecard import scorecard
from brain_agent.settings import ROOT, Settings
from brain_agent.ticks import TickStartError, poll_watches, run_tick, write_report


async def tick(args: argparse.Namespace) -> int:
    settings = Settings.from_env(read_only=args.read_only)
    if not settings.agent_key:
        print("BRAIN_AGENT_KEY is not set (get one at https://bitgn.com/me/agents)", file=sys.stderr)
        return 2
    try:
        result = await run_tick(settings, task=args.task, focus=args.focus, max_calls=args.max_calls)
    except TickStartError as exc:
        print(f"could not start a Brain session: {exc}", file=sys.stderr)
        return 1
    print(f"tick {result.stamp} role={result.role} author={result.author}")
    print(json.dumps(write_report(settings, result), ensure_ascii=False, indent=2))
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
    """Tick at least every `--hours`, and early when a watched job finishes.

    Between ticks the watched jobs are polled every `--poll-minutes` with one cheap
    read-only program. One failed tick or poll never stops the loop.
    """
    settings = Settings.from_env(read_only=args.read_only)
    while True:
        try:
            await tick(args)
        except Exception as exc:  # logged, then wait for the next slot
            print(f"tick failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.stdout.flush()
        deadline = time.monotonic() + args.hours * 3600
        while (left := deadline - time.monotonic()) > 0:
            await asyncio.sleep(min(args.poll_minutes * 60, left))
            try:
                fired = await poll_watches(settings)
            except Exception as exc:
                print(f"poll failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if fired:
                print(f"woken by: {', '.join(f'{w['ref']} ({w['state']})' for w in fired)}")
                break


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
            p.add_argument("--hours", type=float, default=1.0, help="longest gap between ticks")
            p.add_argument("--poll-minutes", type=float, default=10.0, help="how often watched jobs are checked")
    p = sub.add_parser("scorecard", help="summarize ticks and check published work in Brain")
    p.add_argument("--offline", action="store_true", help="local logs only, no Brain session")
    p = sub.add_parser("eval", help="compare agent variants on read-only scenarios with an LLM judge")
    p.add_argument(
        "--variant",
        action="append",
        required=True,
        help="name=model or name=model@extra_instructions.md; repeat; the first is the baseline",
    )
    p.add_argument("--case", action="append", choices=["choose", "review", "readout"], help="default: all")
    p.add_argument("--judge", default=os.environ.get("BRAIN_JUDGE_MODEL", "zai:glm-5.3"))
    p.add_argument("--repeat", type=int, default=1)
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
    if args.command == "eval":
        from brain_agent.evals import CASES, run_evals

        text = asyncio.run(
            run_evals(Settings.from_env(), args.variant, args.case or list(CASES), args.judge, args.repeat)
        )
        print(text)
        return
    sys.exit(asyncio.run(loop(args) if args.command == "loop" else tick(args)))
