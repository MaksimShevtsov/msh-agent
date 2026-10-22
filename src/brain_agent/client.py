"""Client for BitGN Brain over its MCP server.

The agent key lives in the MCP URL (``https://brain.bitgn.com/mcp/agt-...``,
Streamable HTTP, no auth headers) and fixes the agent's identity. The server
exposes two tools:

* ``brain_start(task)`` opens a task session and returns its handle,
  ``next_seq=1``, the agent identity, workspace index and SDK guide.
* ``brain_run(session, seq, source)`` executes Python (Monty subset) on the
  server with ``brain`` bound to that session.

The client owns ``seq`` so the model never has to: each response carries
``next_seq``, an uncertain run is resent with the *same* seq and source (the
server returns the accepted receipt instead of executing twice), and a
recorded error still consumes its seq. Runs are serialized: a model that
calls the tool twice in one turn would otherwise send both with the same seq.

Both tools answer with one text block holding the JSON envelope
``{code, output, data, notices}`` that the HTTP API uses too.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.mcp import MCPToolset

DEFAULT_BASE_URL = "https://brain.bitgn.com"


class BrainError(RuntimeError):
    """The service refused a request or answered with something unreadable."""


@dataclass
class RunResult:
    code: int
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    notices: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def request(self) -> str | None:
        return self.data.get("request")

    @property
    def truncated(self) -> bool:
        return bool(self.data.get("output_truncated"))


def parse_envelope(raw: Any) -> RunResult:
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise BrainError(f"unreadable response: {raw[:300]}") from None
    if not isinstance(raw, dict) or "code" not in raw:
        raise BrainError(f"unexpected response: {str(raw)[:300]}")
    return RunResult(
        code=raw.get("code", 0),
        output=raw.get("output") or "",
        data=raw.get("data") or {},
        notices=raw.get("notices") or [],
    )


@dataclass
class Session:
    handle: str
    next_seq: int = 1


class BrainClient:
    def __init__(self, agent_key: str, base_url: str = DEFAULT_BASE_URL, *, server: Any = None) -> None:
        # The toolset is used as a client only: the model gets our seq-managing
        # wrappers, never the raw brain_start/brain_run tools.
        self._server = server or MCPToolset(f"{base_url}/mcp/{agent_key}", tool_error_behavior="error")
        self._run_lock = asyncio.Lock()

    async def __aenter__(self) -> BrainClient:
        await self._server.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._server.__aexit__(*exc)

    async def _call(self, name: str, args: dict[str, Any]) -> RunResult:
        try:
            raw = await self._server.direct_call_tool(name, args)
        except Exception as exc:
            # A failed program (code != 0) comes back as an MCP tool error whose
            # text is still the full envelope, next_seq included: that is an
            # answer, not a lost response.
            try:
                return parse_envelope(str(exc))
            except BrainError:
                raise BrainError(f"{name}: {type(exc).__name__}: {exc}") from exc
        return parse_envelope(raw)

    async def start(self, task: str) -> tuple[Session | None, RunResult]:
        result = await self._call("brain_start", {"task": task})
        handle = result.data.get("session")
        session = Session(handle, result.data.get("next_seq", 1)) if handle else None
        return session, result

    async def run(self, session: Session, source: str) -> RunResult:
        async with self._run_lock:
            args = {"session": session.handle, "seq": session.next_seq, "source": source}
            try:
                result = await self._call("brain_run", args)
            except BrainError:
                # Uncertain outcome: resend the identical request; an accepted run is
                # returned from its receipt, never executed twice.
                result = await self._call("brain_run", args)
            session.next_seq = result.data.get("next_seq") or session.next_seq + 1
            return result
