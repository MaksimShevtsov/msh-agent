"""What the agent may send to Brain (sandbox specifics live in sandbox_guard).

The checks are textual and therefore best-effort: they stop the model from
*accidentally* doing something expensive or shared-destructive, not a
determined adversary. ``getattr`` is refused so the obvious alias route
(``getattr(brain.sandbox, "sub" + "mit")``) is closed too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Workspace-level verbs and cancelling jobs act on everyone's shared state.
_ALWAYS_BLOCKED = {
    "create_workspace": r"\bcreate_workspace\s*\(",
    "rename_workspace": r"\brename_workspace\s*\(",
    "delete_workspace": r"\bdelete_workspace\s*\(",
    "restore_workspace": r"\brestore_workspace\s*\(",
    "artifacts.revoke": r"\brevoke\s*\(",
    "getattr": r"\bgetattr\s*\(",
}
_SUBMIT = r"\b(submit|cancel)\s*\("
_PUBLISH = r"\b(commit|publish)\s*\("


@dataclass(frozen=True)
class Policy:
    # False: drafts may be staged but never published; the run ends with discard().
    allow_commit: bool = False
    # Full ECOM1 runs go into one shared queue with community depth norms.
    allow_sandbox: bool = False


def check_source(source: str, policy: Policy) -> str | None:
    """Return a refusal reason, or None when the source may be sent."""
    for name, pattern in _ALWAYS_BLOCKED.items():
        if re.search(pattern, source):
            return f"`{name}` is not allowed for this agent."
    if not policy.allow_sandbox and re.search(_SUBMIT, source):
        return (
            "Sandbox submissions are disabled for this agent (shared queue). "
            "Propose the run in your report instead."
        )
    if not policy.allow_commit and re.search(_PUBLISH, source):
        return (
            "This is a read-only run: commit/publish are disabled. "
            "Stage nothing you need to keep; call brain.discard() to finish."
        )
    return None


def closes_session(source: str) -> bool:
    return bool(re.search(r"\b(commit|discard)\s*\(", source))
