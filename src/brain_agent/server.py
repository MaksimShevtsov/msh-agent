"""AG-UI server for chatting with the agent from CopilotKit (`brain-agent serve`).

AG-UI is stateless per request (the client resends the conversation), but a
Brain session holds drafts and a seq counter, so deps live per AG-UI thread
for the lifetime of the process. Open sessions are discarded on shutdown.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime

import logfire
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from brain_agent.agent import BrainDeps, build_agent
from brain_agent.approvals import APPROVAL_TOOL, Approvals
from brain_agent.client import DEFAULT_BASE_URL, BrainClient
from brain_agent.settings import Settings

AG_UI_VERSION = "0.1.10"

CHAT_INSTRUCTIONS = f"""\
This run is an interactive chat with your operator, not a scheduled tick.
Answer the operator directly and briefly. When they give you a goal, carry it
through yourself -- read, decide, draft, publish -- instead of asking about each
step; ask only when a choice is genuinely theirs. End with what you will do next.
No Brain session is open at the start of a conversation: call brain_start(task)
when you first need the workspace, and keep using that session.
Questions about the scheduled loop (is a tick running, what did the last tick do,
which sandbox jobs are yours) go to `loop_status`; it needs no Brain session.
Your notebook only has what finished ticks wrote, so check loop_status for the live state.
Commits and sandbox submissions need the operator's approval: when brain_run
answers "[approval required]", call `{APPROVAL_TOOL}` with the approval_id it gave
and stop; after an approval, resend the identical program. Never claim an action
was approved or done unless the tool results say so."""


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or Settings.from_env()
    agent = build_agent(
        settings.model,
        agent_name=settings.agent_name,
        state_dir=settings.state_dir,
        policy=settings.policy,
        run_budget_usd=settings.run_budget_usd,
        with_loop_status=True,
        advisor_model=settings.advisor_model,
    )
    client = BrainClient(settings.agent_key, os.environ.get("BRAIN_URL", DEFAULT_BASE_URL))
    threads: dict[str, BrainDeps] = {}
    approvals = Approvals()

    def deps_for(thread_id: str) -> BrainDeps:
        if thread_id not in threads:
            stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
            threads[thread_id] = settings.deps(
                client, session=None, log_path=settings.state_dir / "chat" / f"{stamp}-{thread_id[:8]}.jsonl"
            )
            threads[thread_id].approvals = approvals
            threads[thread_id].max_calls = 200
        deps = threads[thread_id]
        deps.calls = 0  # the call budget is per operator turn in chat
        return deps

    async def run(request: Request) -> Response:
        run_input = AGUIAdapter.build_run_input(await request.body())
        return await AGUIAdapter.dispatch_request(
            request,
            agent=agent,
            deps=deps_for(run_input.thread_id),
            output_type=str,
            instructions=CHAT_INSTRUCTIONS,
            # CopilotKit 1.51 ships @ag-ui/client 0.0.42, which predates REASONING_*
            # events and aborts the stream on one; < 0.1.11 emits THINKING_* instead.
            ag_ui_version=AG_UI_VERSION,
        )

    async def approval(request: Request) -> Response:
        pending = approvals.pending.get(request.path_params["approval_id"])
        if pending is None:
            return JSONResponse({"error": "unknown or already answered"}, status_code=404)
        return JSONResponse(asdict(pending))

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True, "threads": len(threads)})

    @asynccontextmanager
    async def lifespan(_: Starlette):
        async with client:
            yield
            for deps in threads.values():
                if deps.session is not None:
                    try:
                        await client.run(deps.session, "brain.discard()")
                    except Exception:
                        pass

    app = Starlette(
        routes=[
            Route("/", run, methods=["POST"]),
            Route("/approvals/{approval_id}", approval),
            Route("/health", health),
        ],
        lifespan=lifespan,
    )
    logfire.instrument_starlette(app)
    return app


def serve(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
