import { NextRequest } from "next/server";

const agentUrl = (process.env.AGENT_URL ?? "http://localhost:8000/").replace(/\/$/, "");

// The server's own copy of what will run -- never the model's tool arguments.
export async function GET(_req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[0-9a-f]{16}$/.test(id)) {
    return Response.json({ error: "bad approval id" }, { status: 400 });
  }
  const upstream = await fetch(`${agentUrl}/approvals/${id}`, { cache: "no-store" });
  return Response.json(await upstream.json(), { status: upstream.status });
}
