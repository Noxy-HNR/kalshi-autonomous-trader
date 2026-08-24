"""Structured output contracts for the LLM.

An LLM asked for a number will always produce one. It will produce one when it
has no information, when the question is malformed, and when it has
misunderstood the market entirely -- and the number will look exactly like a
number produced from genuine reasoning.

The defence is to demand structure that is hard to fake accidentally, and to
reject anything that does not parse. A model that cannot name its evidence, state
what would prove it wrong, or keep its probability inside [0, 1] has not produced
an opinion worth acting on, whatever the prose around it says.

Everything here is a strict pydantic model. There is no lenient parsing path and
no "best effort" fallback: an output that does not validate is discarded and
logged, and the quantitative pipeline proceeds without it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "SYSTEM_PROMPT",
    "LLMRejection",
    "MarketThesis",
    "build_market_prompt",
]


class MarketThesis(BaseModel):
    """What the LLM is asked to produce, and all it is allowed to produce.

    Note what is absent: no order, no size, no ticker to trade, no instruction of
    any kind. The advisor consumes this as an opinion about *probability*, and
    everything downstream treats it as one input among several.
    """

    model_config = {"extra": "forbid"}

    probability: float = Field(
        description="Probability the market resolves YES, in [0, 1].",
    )
    confidence: float = Field(
        description="How much weight the model itself places on that estimate, in [0, 1].",
    )
    thesis: str = Field(
        min_length=20,
        max_length=2000,
        description="The reasoning, in prose.",
    )
    key_facts: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Specific facts the estimate rests on.",
    )
    expected_catalyst: str | None = Field(
        default=None,
        max_length=500,
        description="What would move this market, and roughly when.",
    )
    invalidation: list[str] = Field(
        min_length=1,
        max_length=6,
        description="Observations that would prove this thesis wrong.",
    )
    alternative_interpretation: str | None = Field(
        default=None,
        max_length=1000,
        description="The strongest case against the thesis.",
    )
    risks: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="What could go wrong with a position taken on this view.",
    )
    information_sufficiency: Literal["sufficient", "thin", "insufficient"] = Field(
        description="Whether the supplied context supports an estimate at all.",
    )

    @field_validator("probability", "confidence")
    @classmethod
    def _within_unit_interval(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"must be within [0, 1], got {value}")
        return value

    @field_validator("key_facts", "invalidation", "risks")
    @classmethod
    def _no_empty_entries(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v and v.strip()]
        if not cleaned and values:
            raise ValueError("list entries cannot all be blank")
        return cleaned

    @model_validator(mode="after")
    def _confidence_must_match_stated_sufficiency(self) -> MarketThesis:
        """A model that says it lacks information may not also be confident.

        This catches the most common failure: an eloquent answer with a high
        confidence attached to an estimate the model has just admitted it cannot
        support. If it disagrees with itself, we take the pessimistic half.
        """
        if self.information_sufficiency == "insufficient" and self.confidence > 0.2:
            raise ValueError(
                "declared the information insufficient but reported confidence "
                f"{self.confidence}; these cannot both be true"
            )
        if self.information_sufficiency == "thin" and self.confidence > 0.6:
            raise ValueError(
                "declared the information thin but reported confidence "
                f"{self.confidence}"
            )
        return self

    @property
    def is_actionable(self) -> bool:
        """Whether this opinion is worth passing to the sizing layer at all."""
        return self.information_sufficiency == "sufficient" and self.confidence >= 0.5


class LLMRejection(Exception):
    """The model's output was not usable, with the reason recorded.

    Raised rather than returned so that a caller cannot accidentally treat a
    rejected thesis as an accepted one.
    """

    def __init__(self, reason: str, raw: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


SYSTEM_PROMPT = """You are a probability analyst for prediction markets. You \
estimate how likely an event is; you do not place trades and you have no ability \
to do so.

Rules:
- Reply with a single JSON object and nothing else. No prose before or after.
- `probability` is your estimate that the market resolves YES, between 0 and 1.
- `confidence` is how much you trust your own estimate, between 0 and 1.
- If the context does not support an estimate, set `information_sufficiency` to \
"insufficient" and `confidence` at or below 0.2. Saying so is a correct answer, \
not a failure.
- `key_facts` must be specific things from the context, not restatements of the \
question.
- `invalidation` must be observations that would show you were wrong.
- Do not anchor on the market price. It is provided so you can say whether you \
disagree with it, not so you can reproduce it.
"""


def build_market_prompt(
    *,
    ticker: str,
    title: str,
    rules: str | None,
    market_price: float | None,
    close_time: str | None,
    news: list[dict[str, str]] | None = None,
    features: dict[str, float] | None = None,
) -> str:
    """Assemble the context the model is allowed to see.

    Deliberately excluded: account balance, position sizes, equity, credentials,
    and anything else about *us*. The model is asked about the world, not about
    our book, and it has no reason to know how much money is at stake.
    """
    parts = [
        f"Market: {ticker}",
        f"Question: {title}",
    ]
    if rules:
        parts.append(f"Resolution rules: {rules[:1500]}")
    if close_time:
        parts.append(f"Closes: {close_time}")
    if market_price is not None:
        parts.append(
            f"Current market price for YES: {market_price:.4f} "
            f"(implied probability {market_price:.1%})"
        )
    if features:
        readable = ", ".join(f"{k}={v:.4g}" for k, v in sorted(features.items())[:20])
        parts.append(f"Market statistics: {readable}")
    if news:
        parts.append("Recent relevant reporting:")
        for item in news[:8]:
            published = item.get("published", "unknown time")
            parts.append(f"  - [{published}] {item.get('title', '')}")
            summary = item.get("summary")
            if summary:
                parts.append(f"    {summary[:300]}")
    parts.append(
        "\nRespond with a single JSON object matching the required schema. "
        "If the context above does not support an estimate, say so."
    )
    return "\n".join(parts)
