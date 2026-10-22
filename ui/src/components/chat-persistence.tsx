"use client";

import { UseAgentUpdate, useAgent } from "@copilotkit/react-core/v2";
import { useEffect, useRef } from "react";

import { loadMessages, loadThreadId, saveMessages } from "@/lib/chat-storage";

// Restores the conversation after a refresh and saves it on every change.
export function ChatPersistence() {
  const { agent } = useAgent({ agentId: "msh_agent", updates: [UseAgentUpdate.OnMessagesChanged] });
  const restored = useRef(false);

  useEffect(() => {
    if (restored.current) return;
    restored.current = true;
    agent.threadId = loadThreadId();
    const saved = loadMessages<(typeof agent.messages)[number]>();
    if (saved.length && agent.messages.length === 0) agent.setMessages(saved);
  }, [agent]);

  useEffect(() => {
    if (restored.current && agent.messages.length) saveMessages(agent.messages);
  }, [agent, agent.messages]);

  return null;
}
