"""Operator approval for commit and sandbox submit in chat runs.

The approval is a CopilotKit human-in-the-loop *frontend* tool. Its result
reaches the server as a tool-return part in the message history the client
sends -- the model can request it but never author it. The server binds each
request to the exact program and session (``approval_id``), keeps what it will
execute (source, diff) itself, and the UI shows that server copy, not the
model's arguments. An approval is spent once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart

APPROVAL_TOOL = "confirm_brain_action"


def approval_id(session: str, source: str) -> str:
    return hashlib.sha256(f"{session}\0{source}".encode()).hexdigest()[:16]


@dataclass
class PendingApproval:
    approval_id: str
    session: str
    action: str
    source: str
    diff: str


@dataclass
class Approvals:
    pending: dict[str, PendingApproval] = field(default_factory=dict)
    spent: set[str] = field(default_factory=set)

    def request(self, session: str, source: str, action: str, diff: str) -> str:
        key = approval_id(session, source)
        self.pending[key] = PendingApproval(key, session, action, source, diff)
        return key

    def decision(self, messages: Sequence[ModelMessage], session: str, source: str) -> bool | None:
        """True/False once the operator answered for this exact program, else None."""
        key = approval_id(session, source)
        if key in self.spent or key not in self.pending:
            return None
        for message in reversed(messages):
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == APPROVAL_TOOL:
                    answer = _parse(part.content)
                    if answer.get("approval_id") == key:
                        self.pending.pop(key)
                        self.spent.add(key)
                        return answer.get("approved") is True
        return None


def _parse(content: object) -> dict:
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(str(content))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
