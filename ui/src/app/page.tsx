"use client";

import { useHumanInTheLoop, useRenderToolCall } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";

import { ApprovalCard } from "@/components/approval-card";

export default function Page() {
  // Commit / sandbox submit: the agent calls this frontend tool and waits for the click.
  useHumanInTheLoop({
    name: "confirm_brain_action",
    description:
      "Ask the operator to approve a Brain commit or sandbox submission. Pass the approval_id that brain_run returned.",
    parameters: [{ name: "approval_id", type: "string", description: "approval_id from brain_run", required: true }],
    render: ({ args, respond, status }) => (
      <ApprovalCard approvalId={args.approval_id} status={status} respond={respond} />
    ),
  });

  // Show every program the agent sends to Brain, collapsed.
  useRenderToolCall({
    name: "brain_run",
    parameters: [{ name: "source", type: "string", required: true }],
    render: ({ args, status }) => (
      <details className="my-1 rounded border border-neutral-300 px-2 py-1 text-xs dark:border-neutral-700">
        <summary className="cursor-pointer opacity-80">
          brain_run {status === "complete" ? "✓" : "…"}
        </summary>
        <pre className="mt-1 overflow-auto whitespace-pre-wrap">{args.source}</pre>
      </details>
    ),
  });

  return (
    <main className="mx-auto flex h-screen max-w-4xl flex-col px-4 py-4">
      <header className="mb-3 flex items-baseline justify-between">
        <h1 className="text-lg font-semibold">msh-agent</h1>
        <span className="text-xs opacity-60">BitGN Brain · ECOM1 · commits and sandbox runs need your approval</span>
      </header>
      <CopilotChat
        className="flex-1 overflow-hidden rounded-lg border border-neutral-300 dark:border-neutral-700"
        labels={{
          title: "msh-agent",
          initial: "Hi. Ask me about the ECOM1 workspace, or tell me what to work on.",
        }}
      />
    </main>
  );
}
