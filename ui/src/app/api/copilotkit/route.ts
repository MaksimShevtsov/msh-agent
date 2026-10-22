import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

const agentUrl = process.env.AGENT_URL ?? "http://localhost:8000/";

// Single agent, so no LLM service adapter on this side: the Python AG-UI server does the work.
const runtime = new CopilotRuntime({
  agents: { msh_agent: new HttpAgent({ url: agentUrl }) },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
