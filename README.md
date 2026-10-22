# brain-agent

Our own Pydantic AI agent for [BitGN Brain](https://brain.bitgn.com), the shared
ECOM1 research workspace. BitGN agent config: `pydantic-agent`.

## How it works

Brain is an MCP server (`https://brain.bitgn.com/mcp/<agent key>`, Streamable
HTTP, the key in the URL is the identity) with two tools: `brain_start(task)` and
`brain_run(session, seq, source)`. `brain_run` executes Python (Monty subset) on
the server with `brain` bound to the session -- Brain is code mode already.

`client.py` uses pydantic-ai's `MCPToolset` as a client and owns `session`/`seq`:
it advances `next_seq`, resends an uncertain run unchanged (the server returns the
receipt, never re-executes), and reads failed programs from the MCP error envelope.
The model sees only `brain_run(source)` and `brain_start(task)`.

| Capability | Why |
|---|---|
| `Planning` | keeps one tick on track: orient, pick, contribute, record |
| `Memory(SqliteMemoryStore)` | notebook that survives between ticks (`state/memory.sqlite3`) |
| `SpendLimits` (per-run USD) | brake on a runaway loop |
| `ToolOutputLimits` | spills big Brain outputs to `state/overflow` |
| `SummarizingCompaction` | compacts history at 60 % of the model's window |
| `RepairToolArguments` | fixes malformed tool JSON from small models |
| `PromptInjectionDefender` (pattern tier) | withholds Brain output with planted instructions (`ignore previous…`, fake `<system>` tags); logged as `injection`. The ML tier is off: it withheld 13 % of real outputs |
| `ToolCallJudge` + `CommitEvidence` | loop only: a second model (`BRAIN_COMMIT_JUDGE_MODEL`) reviews each commit against the server's diff; uncertain = blocked |
| `TrajectoryJudge` | loop only: `BRAIN_TRAJECTORY_JUDGE_MODEL` reviews the last ~20k tokens every 6 requests, in the background, and steers off re-polling, repeated failures, budget drift and contradicted claims; logged as `review` (scorecard "steers"). An unavailable judge falls back to all-good (`review_unavailable`) instead of failing the tick |

`policy.py` always refuses workspace-level verbs, `artifacts.revoke` and `getattr`;
`commit`/`publish` and sandbox calls only run when enabled in `.env`. Commits go
through their own `brain_commit` tool (`brain_run` refuses any `.commit`/`.publish`),
so the judge sees every one; a blocked commit counts as a refusal in the scorecard.

`sandbox_guard.py` enforces the shared-queue rules in code: one submission per
tick, only as `job = brain.sandbox.submit(..., trials=N)` with a literal `N`; the
client reads the live queue depth first and refuses a full run (N > 10) above
depth 10 and anything above depth 25. Submitted job ids are captured and written
to `state/jobs.json` on commit; the next tick sees them in its prompt, and
`cancel` is allowed only for ids in that ledger.

## Run

```powershell
copy .env.example .env            # BRAIN_AGENT_KEY, ZAI_API_KEY
docker compose build
docker compose run --rm brain-agent tick --read-only     # one safe tick
docker compose up -d                                     # tick every 3 hours
docker compose logs -f
```

### Chat UI (CopilotKit)

`docker compose up -d chat ui`, then open http://localhost:3000. `chat` runs
`brain-agent serve`: an AG-UI endpoint (`AGUIAdapter`) that keeps one Brain
session per conversation thread. In chat, `commit` and `sandbox.submit` need your
click: `brain_run` answers `[approval required]` with an `approval_id`, the agent
calls the `confirm_brain_action` frontend tool, and the card shows the server's
copy of the program and the draft diff (`GET /approvals/{id}`), not the model's
arguments. The server runs the program only when the conversation history carries
your answer for that exact program and session, and each approval works once.
Chat logs go to `state/chat/`.

### Measuring and improving the agent

- **Tracing:** set `LOGFIRE_TOKEN` in `.env` and every tick and chat turn shows up in
  Logfire as a tree of model requests, tool calls and MCP calls to Brain, with cost.
  Without the token nothing is sent.
- **`brain-agent scorecard`** (`--offline` for logs only) writes `state/scorecard.md`: per
  tick cost, Brain calls, failed programs, timeouts, refusals and commits; then, read-only
  in Brain, how many *other* documents cite each document the agent published (from ticks
  or chat), and the state and score of sandbox jobs its reports point at -- the evidence
  for checking its predictions.
- **`brain-agent eval --variant base=zai:glm-5.3-flash --variant big=zai:glm-5.3`**
  (optionally `name=model@extra_instructions.md` for prompt A/B, `--case choose|review|readout`,
  `--judge MODEL`, `--repeat N`) runs each variant on read-only scenarios from a copy of
  the notebook; the agent drafts instead of committing. Scored by cost, Brain calls,
  failed-program rate, GroundedIds (ids in the draft that it actually saw) and an LLM
  judge (grounded, useful, norms). The first variant is the baseline in the comparison
  table; results land in `state/evals/<stamp>/`. The workspace is live, so compare
  variants from the same eval run only.

Every tick writes `state/runs/<stamp>.jsonl` (each program sent and its output)
and `state/reports/<stamp>.json` (the agent's report). Without Docker:
`uv sync`, `uv run pytest`, `uv run brain-agent tick --read-only`.
