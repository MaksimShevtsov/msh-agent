// Browser-side persistence of the current chat: thread id + AG-UI messages.
const THREAD_KEY = "msh-agent.threadId";
const MESSAGES_KEY = "msh-agent.messages";

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // Storage blocked (private window): the chat still works, it just won't survive a refresh.
  }
}

export function loadThreadId(): string {
  const existing = read(THREAD_KEY);
  if (existing) return existing;
  const fresh = crypto.randomUUID();
  write(THREAD_KEY, fresh);
  return fresh;
}

export function loadMessages<T>(): T[] {
  try {
    const parsed = JSON.parse(read(MESSAGES_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveMessages(messages: unknown[]): void {
  write(MESSAGES_KEY, JSON.stringify(messages));
}

export function startNewChat(): void {
  write(THREAD_KEY, null);
  write(MESSAGES_KEY, null);
}
