"""Local claims shared by this agent's own processes (the loop and the chat).

Both run as the same Brain identity and share one notebook, so both can pick the
same follow-up at the same time. Before working on a document or job they claim
it here; a claim held by the other process (and not yet expired) is refused.
This only dedupes msh-agent against itself -- LANES.md is still where the swarm
learns what it is doing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

TTL_SECONDS = 2 * 3600


class Claims:
    def __init__(self, path: Path, owner: str) -> None:
        self.path = path
        self.owner = owner  # "loop" or "chat:<thread>"

    def _load(self) -> dict[str, dict]:
        try:
            claims = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        now = time.time()
        return {ref: c for ref, c in claims.items() if now - c["t"] < TTL_SECONDS}

    def claim(self, refs: list[str], what: str) -> str:
        claims = self._load()
        held = {ref: claims[ref] for ref in refs if ref in claims and claims[ref]["owner"] != self.owner}
        if held:
            return "Already being worked on by your other process -- pick something else:\n" + "\n".join(
                f"- {ref}: {c['what']} ({c['owner']}, {int((time.time() - c['t']) / 60)} min ago)"
                for ref, c in held.items()
            )
        for ref in refs:
            claims[ref] = {"owner": self.owner, "what": what, "t": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(claims, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"Claimed {', '.join(refs)} for {self.owner}."

    def render(self) -> str:
        claims = self._load()
        if not claims:
            return "(no live claims)"
        return "\n".join(f"- {ref}: {c['what']} ({c['owner']})" for ref, c in claims.items())
