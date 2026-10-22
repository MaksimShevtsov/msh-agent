"""Model routing: z.ai GLM goes to the coding endpoint the subscription covers."""

from __future__ import annotations

import os

from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider

# The SDK default (api/paas/v4) is billed separately; the coding plan covers this one.
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


def _provider(name: str) -> Provider:
    provider = infer_provider(name)
    if name == "zai":
        provider.client.base_url = os.environ.get("ZAI_BASE_URL", "").strip() or ZAI_CODING_BASE_URL
    return provider


def resolve_model(model_id: str) -> Model:
    return infer_model(model_id, provider_factory=_provider)
