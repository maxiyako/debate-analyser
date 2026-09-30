"""Thin google-genai helpers for structured output and grounded search.

Cycle rule: `src.agents` imports `src.briefing`, which imports this module. So
this file must not import `src.agents` or `src.validation` at module level;
retry and redirect helpers are imported inside the functions that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from config import Settings
from src.costs import TRACKER

T = TypeVar("T", bound=BaseModel)


def _client(settings: Settings) -> tuple[Any, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        http_options=types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000)),
    )
    return client, types


def generate_json(
    prompt: str,
    schema: type[T],
    settings: Settings,
    *,
    label: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> T:
    """One structured-output call validated into `schema` (with 429 backoff)."""
    from src.agents import _retry_on_rate_limit

    client, types = _client(settings)

    def _call():
        return client.models.generate_content(
            model=model or settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

    resp = _retry_on_rate_limit(_call)
    TRACKER.add_genai(label, getattr(resp, "usage_metadata", None))
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    return schema.model_validate_json(resp.text)


@dataclass
class GroundedAnswer:
    text: str
    urls: list[str]


def grounded_search(prompt: str, settings: Settings, *, label: str) -> GroundedAnswer:
    """Google-Search-grounded answer plus the real cited URLs.

    Citations are redirect-resolved and filtered only by the hard disinfo/party
    blocklist — no positive allowlist (source tiering arrives in a later plan).
    """
    from src.agents import _resolve_redirect, _retry_on_rate_limit
    from src.validation import is_blocked_source

    client, types = _client(settings)

    def _call():
        return client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

    resp = _retry_on_rate_limit(_call)
    TRACKER.add_genai(label, getattr(resp, "usage_metadata", None))
    urls: list[str] = []
    try:
        meta = resp.candidates[0].grounding_metadata
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None):
                real = _resolve_redirect(web.uri)
                if real not in urls and not is_blocked_source(real):
                    urls.append(real)
    except (AttributeError, IndexError, TypeError):
        pass
    return GroundedAnswer(text=(resp.text or "").strip(), urls=urls)
