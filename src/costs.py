"""Token / USD cost tracking for Vertex Gemini usage.

Pricing is approximate and editable — values are USD per 1M tokens.
Grounding Search has a separate per-query fee not included here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# USD per 1M tokens (approximate; update when Google changes list prices).
_PRICES: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {
        "input": 1.25,
        "cached_input": 0.31,
        "output": 10.0,
    },
}


@dataclass
class _Row:
    label: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CostTracker:
    rows: list[_Row] = field(default_factory=list)

    def reset(self) -> None:
        self.rows.clear()

    def add_crew(self, label: str, crew_output: Any) -> None:
        usage = getattr(crew_output, "token_usage", None)
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        cached = int(getattr(usage, "cached_prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        requests = int(getattr(usage, "successful_requests", 0) or 0)
        self.rows.append(
            _Row(
                label=label,
                prompt_tokens=prompt,
                cached_tokens=cached,
                completion_tokens=completion,
                requests=requests,
            )
        )

    def add_genai(self, label: str, usage_metadata: Any) -> None:
        if usage_metadata is None:
            return
        prompt = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
        cached = int(getattr(usage_metadata, "cached_content_token_count", 0) or 0)
        completion = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        # Merge into existing label row so many grounded_search calls collapse.
        for row in self.rows:
            if row.label == label:
                row.prompt_tokens += prompt
                row.cached_tokens += cached
                row.completion_tokens += completion
                row.requests += 1
                return
        self.rows.append(
            _Row(
                label=label,
                prompt_tokens=prompt,
                cached_tokens=cached,
                completion_tokens=completion,
                requests=1,
            )
        )

    def _estimate_usd(self, model: str) -> float:
        prices = _PRICES.get(model) or _PRICES["gemini-2.5-pro"]
        total = 0.0
        for row in self.rows:
            cached = min(row.cached_tokens, row.prompt_tokens)
            billable_input = max(0, row.prompt_tokens - cached)
            total += billable_input * prices["input"] / 1_000_000
            total += cached * prices["cached_input"] / 1_000_000
            total += row.completion_tokens * prices["output"] / 1_000_000
        return total

    def summary(self, model: str) -> str:
        if not self.rows:
            return "Cost: no LLM usage recorded."

        lines = [f"Cost estimate ({model}; token fees only, excl. Search grounding query fees):"]
        prompt_sum = cached_sum = completion_sum = requests_sum = 0
        for row in self.rows:
            prompt_sum += row.prompt_tokens
            cached_sum += row.cached_tokens
            completion_sum += row.completion_tokens
            requests_sum += row.requests
            lines.append(
                f"  {row.label}: "
                f"in={row.prompt_tokens:,} (cached={row.cached_tokens:,}) "
                f"out={row.completion_tokens:,} "
                f"req={row.requests}"
            )
        usd = self._estimate_usd(model)
        lines.append(
            f"  TOTAL: in={prompt_sum:,} (cached={cached_sum:,}) "
            f"out={completion_sum:,} req={requests_sum} "
            f"≈ ${usd:.4f}"
        )
        return "\n".join(lines)


TRACKER = CostTracker()
