"""The LLM advisory layer.

**The LLM has no authority.** That is a structural fact about this module, not a
policy statement, and it is worth being precise about how it is enforced:

* This module imports no execution engine, no REST client, and no risk
  configuration. There is nothing here it could call to place an order even if
  its output asked to.
* It returns an :class:`Advice` object containing a probability and a rationale.
  It does not return an order, an intent, or a size.
* Its influence on position size is bounded by ``settings.llm.max_influence_pct``
  and applied by :meth:`blend`, which is pure arithmetic over two numbers and
  cannot move a position further than that fraction.
* Every consultation is recorded -- accepted or rejected -- so the question "did
  the model ever suggest this?" is answerable after the fact.

What it cannot bypass, because it never touches them: risk controls, the 1.00x
leverage invariant, position limits, execution controls, kill switches, and the
strategy validation gates. An order originating from an LLM-influenced signal
goes through exactly the same pipeline as any other, and is rejected by the same
checks.

It is also **off by default** and defaults to a local server, so the platform is
never dependent on a paid hosted model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kat.common.ids import new_id
from kat.config.settings import LLMSettings
from kat.db.models import LLMSignal
from kat.db.writer import AsyncWriter
from kat.llm.client import LLMClient, LLMUnavailable
from kat.llm.schemas import LLMRejection, MarketThesis, build_market_prompt
from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = ["Advice", "LLMAdvisor"]


@dataclass(slots=True)
class Advice:
    """An opinion. Not an instruction.

    Contains no ticker to trade, no size, and no side -- deliberately. The caller
    already knows which market it asked about; handing back a tradeable object
    would blur the line this whole module exists to keep sharp.
    """

    signal_id: str
    probability: float
    confidence: float
    accepted: bool
    reason: str
    thesis: MarketThesis | None = None
    latency_ms: float = 0.0
    model: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.accepted and self.thesis is not None and self.thesis.is_actionable


class LLMAdvisor:
    """Consults a language model for a probability opinion, and records it."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        writer: AsyncWriter | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.writer = writer
        self.client = client or LLMClient(settings)
        self.consultations = 0
        self.accepted = 0
        self.rejected = 0

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    # -- consultation --------------------------------------------------------

    async def advise(
        self,
        *,
        ticker: str,
        title: str,
        rules: str | None = None,
        market_price: float | None = None,
        close_time: str | None = None,
        news: list[dict[str, str]] | None = None,
        features: dict[str, float] | None = None,
        decision_id: str | None = None,
        market_type: str = "event",
    ) -> Advice:
        """Ask for an opinion. Never raises; a failure is an unusable Advice.

        The platform must work with the LLM switched off, unreachable, slow, or
        producing nonsense, and the simplest way to guarantee that is for this
        method to have no failure mode the caller must handle.
        """
        signal_id = new_id("llm")

        if not self.settings.enabled:
            return Advice(
                signal_id=signal_id,
                probability=0.5,
                confidence=0.0,
                accepted=False,
                reason="LLM is disabled",
            )

        self.consultations += 1
        prompt = build_market_prompt(
            ticker=ticker,
            title=title,
            rules=rules,
            market_price=market_price,
            close_time=close_time,
            news=news,
            features=features,
        )

        raw: str | None = None
        try:
            thesis, raw, latency_ms = await self.client.complete_thesis(prompt)
        except LLMUnavailable as exc:
            return self._record_rejection(
                signal_id, f"LLM unavailable: {exc}", ticker, market_type,
                prompt, None, decision_id,
            )
        except LLMRejection as exc:
            return self._record_rejection(
                signal_id, f"output rejected: {exc.reason}", ticker, market_type,
                prompt, exc.raw, decision_id,
            )
        except Exception as exc:
            log.error("llm.unexpected_failure", ticker=ticker, error=str(exc))
            return self._record_rejection(
                signal_id, f"unexpected failure: {type(exc).__name__}", ticker,
                market_type, prompt, raw, decision_id,
            )

        advice = Advice(
            signal_id=signal_id,
            probability=thesis.probability,
            confidence=thesis.confidence,
            accepted=True,
            reason=(
                "accepted"
                if thesis.is_actionable
                else f"accepted but not actionable: {thesis.information_sufficiency}"
            ),
            thesis=thesis,
            latency_ms=latency_ms,
            model=self.settings.model,
        )
        self.accepted += 1
        self._persist(advice, ticker, market_type, prompt, raw, decision_id)
        log.info(
            "llm.advice",
            signal_id=signal_id, ticker=ticker,
            probability=round(thesis.probability, 4),
            confidence=round(thesis.confidence, 4),
            sufficiency=thesis.information_sufficiency,
            actionable=thesis.is_actionable,
            latency_ms=round(latency_ms, 1),
        )
        return advice

    # -- influence -----------------------------------------------------------

    def blend(self, quantitative_probability: float, advice: Advice) -> float:
        """Combine the model's view with the quantitative estimate.

        A weighted average where the LLM's weight is capped at
        ``max_influence_pct`` and further scaled by its own stated confidence.
        With the default cap of 0.25 and a confidence of 0.8, the LLM moves the
        final probability by at most 20% of the distance between the two views.

        Deliberately linear and dull. A cleverer scheme would be harder to reason
        about, and the thing that matters most here is being able to state
        exactly how much influence the model has.
        """
        if not advice.usable:
            return quantitative_probability

        cap = float(self.settings.max_influence_pct)
        weight = cap * max(0.0, min(1.0, advice.confidence))
        blended = (1.0 - weight) * quantitative_probability + weight * advice.probability

        log.debug(
            "llm.blended",
            signal_id=advice.signal_id,
            quantitative=round(quantitative_probability, 4),
            llm=round(advice.probability, 4),
            weight=round(weight, 4),
            result=round(blended, 4),
        )
        return max(0.0, min(1.0, blended))

    # -- recording -----------------------------------------------------------

    def _record_rejection(
        self,
        signal_id: str,
        reason: str,
        ticker: str,
        market_type: str,
        prompt: str,
        raw: str | None,
        decision_id: str | None,
    ) -> Advice:
        self.rejected += 1
        advice = Advice(
            signal_id=signal_id,
            probability=0.5,
            confidence=0.0,
            accepted=False,
            reason=reason,
            model=self.settings.model,
        )
        self._persist(advice, ticker, market_type, prompt, raw, decision_id)
        log.warning("llm.rejected", signal_id=signal_id, ticker=ticker, reason=reason)
        return advice

    def _persist(
        self,
        advice: Advice,
        ticker: str,
        market_type: str,
        prompt: str,
        raw: str | None,
        decision_id: str | None,
    ) -> None:
        """Record every consultation, used or not.

        Rejections are recorded too: a model that produces unusable output half
        the time is a fact worth being able to discover, and it is invisible if
        only the accepted answers are stored.
        """
        if self.writer is None:
            return
        self.writer.enqueue(
            LLMSignal(
                signal_id=advice.signal_id,
                provider="openai_compatible",
                model=self.settings.model,
                prompt_id=None,
                market_type=market_type,
                ticker=ticker,
                context={"prompt": prompt[:8000], "base_url": self.settings.base_url},
                raw_output=(raw or "")[:8000] or None,
                parsed_output=(
                    advice.thesis.model_dump(mode="json") if advice.thesis else None
                ),
                probability=advice.probability if advice.accepted else None,
                confidence=advice.confidence if advice.accepted else None,
                accepted=advice.accepted,
                rejection_reason=None if advice.accepted else advice.reason,
                decision_id=decision_id,
                latency_ms=advice.latency_ms or None,
            )
        )

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "model": self.settings.model,
            "base_url": self.settings.base_url,
            "max_influence_pct": str(self.settings.max_influence_pct),
            "consultations": self.consultations,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": (
                round(self.accepted / self.consultations, 4) if self.consultations else None
            ),
        }
