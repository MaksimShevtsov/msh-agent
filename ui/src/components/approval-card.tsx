"use client";

import { useEffect, useState } from "react";

type Pending = { approval_id: string; session: string; action: string; source: string; diff: string };

export interface ApprovalCardProps {
  approvalId?: string;
  status: "inProgress" | "executing" | "complete";
  respond?: (response: string) => void;
}

// Shows the server's copy of the program (and diff) the approval unlocks, then
// answers the frontend tool. The server only honours an answer for that exact program.
export function ApprovalCard({ approvalId, status, respond }: ApprovalCardProps) {
  const [pending, setPending] = useState<Pending | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [decision, setDecision] = useState<"approved" | "rejected" | null>(null);

  useEffect(() => {
    if (!approvalId) return;
    fetch(`/api/approvals/${approvalId}`, { cache: "no-store" })
      .then(async (r) => (r.ok ? setPending(await r.json()) : setError((await r.json()).error ?? r.statusText)))
      .catch((e) => setError(String(e)));
  }, [approvalId]);

  const answer = (approved: boolean) => {
    setDecision(approved ? "approved" : "rejected");
    respond?.(JSON.stringify({ approval_id: approvalId, approved }));
  };

  return (
    <div className="my-3 rounded-lg border border-amber-500/60 bg-amber-50 p-4 text-sm text-neutral-900 dark:bg-amber-950/40 dark:text-neutral-100">
      <div className="mb-2 font-semibold">
        Approval needed: {pending?.action ?? "…"} {pending && <span className="font-normal">· session {pending.session}</span>}
      </div>
      {error && <div className="text-red-700 dark:text-red-400">Cannot load the request: {error}</div>}
      {pending && (
        <>
          <div className="mb-1 text-xs uppercase tracking-wide opacity-70">Program</div>
          <pre className="mb-3 max-h-64 overflow-auto rounded bg-neutral-900 p-2 text-xs text-neutral-100 whitespace-pre-wrap">
            {pending.source}
          </pre>
          {pending.diff && (
            <>
              <div className="mb-1 text-xs uppercase tracking-wide opacity-70">Draft diff to publish</div>
              <pre className="mb-3 max-h-96 overflow-auto rounded bg-neutral-900 p-2 text-xs text-neutral-100 whitespace-pre-wrap">
                {pending.diff}
              </pre>
            </>
          )}
        </>
      )}
      {decision ? (
        <div className="font-medium">{decision === "approved" ? "✅ Approved" : "✋ Rejected"}</div>
      ) : (
        status === "executing" && (
          <div className="flex gap-2">
            <button
              disabled={!pending}
              onClick={() => answer(true)}
              className="rounded bg-emerald-700 px-4 py-2 font-semibold text-white disabled:opacity-40"
            >
              Approve
            </button>
            <button onClick={() => answer(false)} className="rounded border border-neutral-400 px-4 py-2 font-semibold">
              Reject
            </button>
          </div>
        )
      )}
    </div>
  );
}
