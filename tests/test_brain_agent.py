import json
from decimal import Decimal
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from brain_agent.agent import BrainDeps, build_agent
from brain_agent.approvals import APPROVAL_TOOL, Approvals
from brain_agent.client import BrainClient, Session
from brain_agent.loop_status import loop_status
from brain_agent.scorecard import collect_ticks, published_paths
from brain_agent.policy import Policy, check_source
from brain_agent.sandbox_guard import JobLedger, SandboxRules, SubmitPlan, check_cancel, check_queue, plan_submit


class FakeBrainServer:
    """Stands in for the MCP toolset: records calls, answers with Brain envelopes."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_first = fail_first
        self.program_error_next = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def direct_call_tool(self, name: str, args: dict):
        self.calls.append((name, dict(args)))
        if self.fail_first:
            self.fail_first = False
            raise ConnectionError("lost response")
        if self.program_error_next:
            self.program_error_next = False
            # What fastmcp raises for a Brain run whose Python failed.
            raise RuntimeError(json.dumps({"code": 2, "output": "Traceback ...", "data": {"next_seq": args["seq"] + 1}}))
        if name == "brain_start":
            return json.dumps({"code": 0, "output": "Started s1", "data": {"session": "s1", "next_seq": 1, "role": "user"}})
        source = args["source"]
        if source == "print(len(brain.sandbox.queue()))":
            output = str(self.queue_depth)
        elif "SUBMITTED_JOB" in source:
            output = "SUBMITTED_JOB jabc123"
        else:
            output = "AGENTS.md text"
        return {"code": 0, "output": output, "data": {"next_seq": args["seq"] + 1}, "notices": []}

    queue_depth = 5


def test_policy_blocks_shared_destructive_and_gated_calls():
    read_only = Policy()
    assert check_source("print(brain.read('d1'))", read_only) is None
    assert check_source("brain.commit('x')", read_only)
    assert check_source("brain.sandbox.submit('d1', revision=1)", read_only)
    assert check_source("brain.delete_workspace('w1')", Policy(allow_commit=True, allow_sandbox=True))
    assert check_source("getattr(brain, 'commit')()", Policy(allow_commit=True))
    assert check_source("brain.commit('x')", Policy(allow_commit=True)) is None


async def test_client_owns_seq_and_retries_with_the_same_seq():
    server = FakeBrainServer()
    async with BrainClient("agt-x", server=server) as client:
        session, _ = await client.start("t")
        assert session == Session("s1", 1)
        server.fail_first = True
        await client.run(session, "print(1)")
        server.program_error_next = True
        failed = await client.run(session, "1/0")
        await client.run(session, "print(2)")
    assert failed.code == 2 and "Traceback" in failed.output
    runs = [args for name, args in server.calls if name == "brain_run"]
    # uncertain run resent unchanged; a failed program still consumes its seq
    assert [r["seq"] for r in runs] == [1, 1, 2, 3]
    assert runs[0] == runs[1]


async def test_agent_runs_code_on_brain_and_refuses_commit_in_read_only(tmp_path: Path):
    step = {"n": 0}

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step["n"] += 1
        if step["n"] == 1:
            return ModelResponse(parts=[ToolCallPart("brain_run", {"source": "print(brain.read('d1'))"})])
        if step["n"] == 2:
            return ModelResponse(parts=[ToolCallPart("brain_run", {"source": "brain.commit('x')"})])
        last = messages[-1].parts[0].content
        assert "refused by operator policy" in last
        return ModelResponse(
            parts=[ToolCallPart("final_result", {"summary": "read AGENTS.md", "committed": False, "next_step": "distill"})]
        )

    server = FakeBrainServer()
    agent = build_agent("test", agent_name="t", state_dir=tmp_path, policy=Policy(), run_budget_usd=Decimal("1"))
    async with BrainClient("agt-x", server=server) as client:
        deps = make_deps(client, Policy(), tmp_path)
        result = await agent.run("go", deps=deps, model=FunctionModel(model))
    assert result.output.committed is False
    assert [args["source"] for _, args in server.calls] == ["print(brain.read('d1'))"]
    assert deps.session == Session("s1", 2)


def make_deps(client: BrainClient, policy: Policy, tmp_path: Path) -> BrainDeps:
    return BrainDeps(
        client=client,
        policy=policy,
        session=Session("s1"),
        log_path=tmp_path / "log.jsonl",
        ledger=JobLedger(tmp_path / "jobs.json"),
    )


SUBMIT = 'job = brain.sandbox.submit("d9", revision=2, trials={n}, name="x")'


def test_submit_shape_and_queue_rules():
    rules = SandboxRules()
    assert plan_submit("print(1)") is None
    assert isinstance(plan_submit("brain.sandbox.submit('d9', revision=2, trials=1)"), str)  # not assigned
    assert isinstance(plan_submit('job = brain.sandbox.submit("d9", revision=2)'), str)  # no literal trials
    assert isinstance(plan_submit(SUBMIT.format(n=1) + "\n" + SUBMIT.format(n=1)), str)  # twice
    smoke, full = plan_submit(SUBMIT.format(n=10)), plan_submit(SUBMIT.format(n=100))
    assert isinstance(smoke, SubmitPlan) and isinstance(full, SubmitPlan)
    assert check_queue(full, 10, rules) is None
    assert check_queue(full, 11, rules)  # full runs wait for a shallow queue
    assert check_queue(smoke, 25, rules) is None
    assert check_queue(smoke, 26, rules)  # nothing above the hard limit


def test_approval_is_bound_to_the_exact_program_and_spent_once():
    approvals = Approvals()
    key = approvals.request("s1", "brain.commit('x')", "commit", "diff")
    answer = [ModelRequest(parts=[ToolReturnPart(APPROVAL_TOOL, json.dumps({"approval_id": key, "approved": True}))])]
    assert approvals.decision(answer, "s1", "brain.commit('y')") is None  # different program
    assert approvals.decision(answer, "s1", "brain.commit('x')") is True
    assert approvals.decision(answer, "s1", "brain.commit('x')") is None  # spent
    key = approvals.request("s1", "brain.commit('z')", "commit", "diff")
    no = [ModelRequest(parts=[ToolReturnPart(APPROVAL_TOOL, {"approval_id": key, "approved": False})])]
    assert approvals.decision(no, "s1", "brain.commit('z')") is False


async def test_chat_commit_waits_for_approval(tmp_path: Path):
    replies: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("brain_run", {"source": "brain.commit('x')"})])
        replies.append(messages[-1].parts[0].content)
        return ModelResponse(
            parts=[ToolCallPart("final_result", {"summary": "s", "committed": False, "next_step": "n"})]
        )

    server = FakeBrainServer()
    policy = Policy(allow_commit=True)
    agent = build_agent("test", agent_name="t", state_dir=tmp_path, policy=policy, run_budget_usd=Decimal("1"))
    async with BrainClient("agt-x", server=server) as client:
        deps = make_deps(client, policy, tmp_path)
        deps.approvals = Approvals()
        await agent.run("go", deps=deps, model=FunctionModel(model))
    assert "[approval required]" in replies[0] and APPROVAL_TOOL in replies[0]
    (pending,) = deps.approvals.pending.values()
    assert pending.source == "brain.commit('x')" and pending.action == "commit"
    sent = [args["source"] for name, args in server.calls if name == "brain_run"]
    assert sent == ["print(brain.diff())"]  # the diff was fetched for the operator; no commit sent


async def test_chat_commit_runs_once_after_the_operator_approves(tmp_path: Path):
    commit = "brain.commit('x')"
    approvals = Approvals()
    key = approvals.request("s1", commit, "commit", "diff")
    history = [
        ModelRequest.user_text_prompt("commit it"),
        ModelResponse(parts=[ToolCallPart(APPROVAL_TOOL, {"approval_id": key}, tool_call_id="c1")]),
        ModelRequest(
            parts=[ToolReturnPart(APPROVAL_TOOL, json.dumps({"approval_id": key, "approved": True}), tool_call_id="c1")]
        ),
    ]
    step = {"n": 0}

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step["n"] += 1
        if step["n"] <= 2:  # resend after approval, then try to replay the spent approval
            return ModelResponse(parts=[ToolCallPart("brain_run", {"source": commit})])
        return ModelResponse(
            parts=[ToolCallPart("final_result", {"summary": "s", "committed": True, "next_step": "n"})]
        )

    server = FakeBrainServer()
    policy = Policy(allow_commit=True)
    agent = build_agent("test", agent_name="t", state_dir=tmp_path, policy=policy, run_budget_usd=Decimal("1"))
    async with BrainClient("agt-x", server=server) as client:
        deps = make_deps(client, policy, tmp_path)
        deps.approvals = approvals
        await agent.run(message_history=history, deps=deps, model=FunctionModel(model))
    sent = [args["source"] for name, args in server.calls if name == "brain_run"]
    assert sent == [commit]  # executed once; the session closed, so no replay reached the server
    assert key in approvals.spent


def test_loop_status_reports_a_running_tick_and_the_last_report(tmp_path: Path):
    import os

    (tmp_path / "reports").mkdir()
    (tmp_path / "runs").mkdir()
    report = tmp_path / "reports" / "2026-10-22T100000Z.json"
    report.write_text(json.dumps({"summary": "reviewed d3054", "committed": True, "next_step": "check smoke"}))
    os.utime(report, (1_000, 1_000))
    log = tmp_path / "runs" / "2026-10-22T120000Z.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"t": 2_000, "kind": "run", "code": 0, "source": "print(brain.read('d1'))\nmore"},
                {"t": 2_010, "kind": "error", "source": "print(brain.search('w1', 'x'))", "error": "4-second limit"},
            ]
        )
    )
    JobLedger(tmp_path / "jobs.json").add([{"job": "jabc", "trials": 10, "session": "s9"}])

    text = loop_status(tmp_path)
    assert "IN PROGRESS" in text and "1 programs, 1 service errors" in text
    assert "print(brain.read('d1'))" in text and "more" not in text  # first line only
    assert "4-second limit" in text
    assert "reviewed d3054" in text and "check smoke" in text
    assert "jabc trials=10" in text


def test_scorecard_counts_ticks_and_finds_what_was_published(tmp_path: Path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "chat").mkdir()
    (tmp_path / "reports" / "T1.json").write_text(
        json.dumps(
            {
                "stamp": "T1",
                "committed": True,
                "calls": 4,
                "usage": "RunUsage(cost=Decimal('0.125'), requests=3)",
                "contribution": "experiments/x/reviews/r.md (d3138r1); roles/review.md entry",
                "next_step": "check jcb45f1aa9c0743ea843416f1f01e46c9",
            }
        )
    )
    (tmp_path / "runs" / "T1.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"t": 0, "kind": "run", "code": 0, "session": "s1", "source": "print(1)"},
                {"t": 60, "kind": "run", "code": 2, "session": "s1", "source": "1/0"},
                {"t": 90, "kind": "error", "session": "s1", "error": "exceeded the 4-second wall-clock limit"},
                {"t": 120, "kind": "refused", "session": "s1", "source": "brain.delete_workspace('w1')"},
            ]
        )
    )
    # A chat session that created a findings note and committed; another that never committed.
    (tmp_path / "chat" / "c.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"kind": "run", "code": 0, "session": "s7", "source": 'brain.create("w1", "findings/n.md", t)'},
                {"kind": "run", "code": 0, "session": "s7", "source": 'brain.commit("note")'},
                {"kind": "run", "code": 0, "session": "s8", "source": 'brain.create("w1", "forum/x/draft.md", t)'},
            ]
        )
    )

    (tick,) = collect_ticks(tmp_path)
    assert (tick.cost, tick.calls, tick.failed_programs, tick.service_errors, tick.timeouts, tick.refusals) == (
        Decimal("0.125"), 4, 1, 1, 1, 1,
    )
    assert tick.minutes == 2 and tick.jobs == ["jcb45f1aa9c0743ea843416f1f01e46c9"]
    assert published_paths(tmp_path) == ["experiments/x/reviews/r.md", "findings/n.md"]


def test_cancel_only_own_literal_jobs():
    assert check_cancel('brain.sandbox.cancel("jabc", reason="x")', {"jabc"}) is None
    assert check_cancel('brain.sandbox.cancel("jzzz", reason="x")', {"jabc"})
    assert check_cancel("brain.sandbox.cancel(job)", {"jabc"})


async def test_submission_is_checked_recorded_on_commit_and_limited_per_tick(tmp_path: Path):
    step = {"n": 0}
    replies: list[str] = []
    sources = [SUBMIT.format(n=100), SUBMIT.format(n=10), "brain.commit('run x')"]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if step["n"]:
            replies.append(messages[-1].parts[0].content)
        if step["n"] < len(sources) + 1:
            step["n"] += 1
            if step["n"] == 4:  # a second submission in the same tick
                return ModelResponse(parts=[ToolCallPart("brain_start", {"task": "t2"})])
            return ModelResponse(parts=[ToolCallPart("brain_run", {"source": sources[step["n"] - 1]})])
        if step["n"] == 4:
            step["n"] += 1
            return ModelResponse(parts=[ToolCallPart("brain_run", {"source": SUBMIT.format(n=1)})])
        return ModelResponse(
            parts=[ToolCallPart("final_result", {"summary": "s", "committed": True, "next_step": "n"})]
        )

    server = FakeBrainServer()
    server.queue_depth = 15
    policy = Policy(allow_commit=True, allow_sandbox=True)
    agent = build_agent("test", agent_name="t", state_dir=tmp_path, policy=policy, run_budget_usd=Decimal("1"))
    async with BrainClient("agt-x", server=server) as client:
        deps = make_deps(client, policy, tmp_path)
        await agent.run("go", deps=deps, model=FunctionModel(model))

    assert "only smoke runs" in replies[0]  # full100 at depth 15 refused before sending
    assert "SUBMITTED_JOB jabc123" in replies[1]
    assert deps.ledger.ids() == {"jabc123"}  # recorded once committed
    assert "At most 1 sandbox submission" in replies[-1]
    sent = [args["source"] for name, args in server.calls if name == "brain_run"]
    assert not any("trials=100" in s for s in sent)
