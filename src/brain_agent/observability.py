"""Logfire tracing: agent runs, model requests, tool calls and MCP calls to Brain.

Sends only when LOGFIRE_TOKEN is set (`send_to_logfire="if-token-present"`), so
local runs and tests stay offline. Traces include prompts, tool arguments and
workspace text -- the same data the run logs keep.
"""

from __future__ import annotations

import os

import logfire

_configured = False


def configure(service: str) -> None:
    """Call once per process, before agents are built or apps are created."""
    global _configured
    if _configured:
        return
    logfire.configure(
        send_to_logfire="if-token-present",
        service_name=service,
        service_version=os.environ.get("BRAIN_AGENT_VERSION", "0.1.0"),
        environment=os.environ.get("BRAIN_ENVIRONMENT", "local"),
        console=False,
    )
    logfire.instrument_pydantic_ai()
    # MCP calls to Brain need no instrument_mcp(): mcp 2 emits its own OTel spans.
    _configured = True
