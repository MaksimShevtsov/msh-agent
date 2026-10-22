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

`policy.py` always refuses workspace-level verbs, `artifacts.revoke` and `getattr`;
`commit`/`publish` and sandbox calls only run when enabled in `.env`.

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

Every tick writes `state/runs/<stamp>.jsonl` (each program sent and its output)
and `state/reports/<stamp>.json` (the agent's report). Without Docker:
`uv sync`, `uv run pytest`, `uv run brain-agent tick --read-only`.
