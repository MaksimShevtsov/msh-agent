"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { useEffect, useState } from "react";

import { loadThreadId } from "@/lib/chat-storage";

// The thread id lives in localStorage so a page refresh continues the same
// conversation -- and the server keeps that thread's Brain session.
export function Providers({ children }: { children: React.ReactNode }) {
  const [threadId, setThreadId] = useState<string | null>(null);
  useEffect(() => setThreadId(loadThreadId()), []);
  if (!threadId) return null;
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="msh_agent" threadId={threadId} showDevConsole={false}>
      {children}
    </CopilotKit>
  );
}
