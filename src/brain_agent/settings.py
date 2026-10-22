"""Settings from the environment (.env), shared by `tick`, `loop` and `serve`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from brain_agent.agenda import Agenda
from brain_agent.agent import BrainDeps
from brain_agent.claims import Claims
from brain_agent.client import BrainClient, Session
from brain_agent.policy import Policy
from brain_agent.sandbox_guard import JobLedger, SandboxRules

ROOT = Path(__file__).resolve().parents[2]


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    agent_key: str
    agent_name: str
    model: str
    advisor_model: str | None
    commit_judge_model: str | None
    state_dir: Path
    policy: Policy
    rules: SandboxRules
    run_budget_usd: Decimal

    @classmethod
    def from_env(cls, *, read_only: bool = False) -> Settings:
        return cls(
            agent_key=os.environ.get("BRAIN_AGENT_KEY", "").strip(),
            agent_name=os.environ.get("BRAIN_AGENT_NAME", "msh-agent"),
            model=os.environ.get("BRAIN_MODEL", "zai:glm-5.3-flash"),
            advisor_model=os.environ.get("BRAIN_ADVISOR_MODEL", "").strip() or None,
            commit_judge_model=os.environ.get("BRAIN_COMMIT_JUDGE_MODEL", "zai:glm-5.3").strip() or None,
            state_dir=Path(os.environ.get("BRAIN_STATE_DIR", ROOT / "state")),
            policy=Policy(
                allow_commit=_flag("BRAIN_ALLOW_COMMIT") and not read_only,
                allow_sandbox=_flag("BRAIN_ALLOW_SANDBOX") and not read_only,
            ),
            rules=SandboxRules(
                smoke_max=int(os.environ.get("BRAIN_SANDBOX_SMOKE_MAX", "10")),
                full_depth_max=int(os.environ.get("BRAIN_SANDBOX_FULL_DEPTH_MAX", "10")),
                hard_depth_max=int(os.environ.get("BRAIN_SANDBOX_HARD_DEPTH_MAX", "25")),
            ),
            run_budget_usd=Decimal(os.environ.get("BRAIN_RUN_BUDGET_USD", "3")),
        )

    @property
    def ledger(self) -> JobLedger:
        return JobLedger(self.state_dir / "jobs.json")

    @property
    def agenda(self) -> Agenda:
        return Agenda(self.state_dir / "agenda.json")

    def deps(
        self, client: BrainClient, *, session: Session | None, log_path: Path, owner: str = "loop"
    ) -> BrainDeps:
        return BrainDeps(
            client=client,
            policy=self.policy,
            session=session,
            log_path=log_path,
            ledger=self.ledger,
            rules=self.rules,
            claims=Claims(self.state_dir / "claims.json", owner),
        )
