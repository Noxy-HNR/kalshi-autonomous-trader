"""OpenAI-compatible client, pointed at a local server by default.

The default target is ``http://127.0.0.1:11434/v1`` -- Ollama, llama.cpp and LM
Studio all speak this dialect -- so the platform can use a language model without
depending on a paid hosted one. A hosted endpoint works by changing the base URL
and supplying a key.

Degradation is the whole design here. If no server is listening, if it is slow,
or if it returns something that is not the schema, the caller gets a clear
failure and the quantitative pipeline carries on unchanged. Nothing in this
module is on a path that must succeed.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
import orjson

from kat.config.settings import LLMSettings
from kat.llm.schemas import SYSTEM_PROMPT, LLMRejection, MarketThesis
from kat.observability.logging import get_logger
from kat.observability.metrics import metrics

log = get_logger(__name__)

__all__ = ["LLMClient", "LLMUnavailable", "extract_json"]


class LLMUnavailable(RuntimeError):
    """No model could be reached. Distinct from a model that answered badly."""


#: Matches the first balanced-looking JSON object in a response. Small models
#: wrap JSON in prose or fences however firmly they are told not to, and
#: rejecting an otherwise valid answer over a markdown fence would be pedantry.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Tolerant about *packaging* -- code fences, a sentence of preamble -- and
    strict about *content*: the result must parse as an object, or this raises.
    There is no partial-credit path.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = orjson.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except orjson.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(stripped)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise LLMRejection(f"response contained no valid JSON object: {exc}", text) from exc

    raise LLMRejection("response contained no JSON object at all", text)


class LLMClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.settings.api_key:
                headers["Authorization"] = f"Bearer {self.settings.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url.rstrip("/"),
                timeout=httpx.Timeout(self.settings.timeout_s, connect=5.0),
                headers=headers,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def available(self) -> bool:
        """Whether a model server is actually reachable.

        Called at start-up so the platform can report the LLM as unavailable
        once, rather than discovering it on every market it looks at.
        """
        if not self.settings.enabled:
            return False
        try:
            response = await self._http().get("/models", timeout=3.0)
            return response.status_code < 500
        except Exception:
            return False

    async def complete_thesis(self, prompt: str) -> tuple[MarketThesis, str, float]:
        """Ask for a thesis and validate it. Returns (thesis, raw, latency_ms).

        Raises :class:`LLMUnavailable` when no model answered and
        :class:`LLMRejection` when one answered with something unusable. The two
        are kept distinct because they mean different things operationally: the
        first is an infrastructure problem, the second is a model problem.
        """
        started = time.monotonic()
        body = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.settings.temperature),
            "max_tokens": self.settings.max_tokens,
            # Honoured by servers that support it and harmlessly ignored by the
            # rest; extract_json copes either way.
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self._http().post("/chat/completions", json=body)
        except httpx.ConnectError as exc:
            raise LLMUnavailable(
                f"no model server at {self.settings.base_url} ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailable(
                f"model did not respond within {self.settings.timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"transport failure: {exc}") from exc

        latency_ms = (time.monotonic() - started) * 1000
        metrics.observe("llm.latency_ms", latency_ms, model=self.settings.model)

        if response.status_code >= 400:
            raise LLMUnavailable(
                f"model server returned {response.status_code}: {response.text[:300]}"
            )

        try:
            payload = orjson.loads(response.content)
            raw = payload["choices"][0]["message"]["content"]
        except (orjson.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMRejection(
                f"response was not a chat completion: {exc}", response.text[:2000]
            ) from exc

        data = extract_json(raw)
        try:
            thesis = MarketThesis.model_validate(data)
        except Exception as exc:
            # A schema failure is a rejection, not a crash. The most common cause
            # is a confident-sounding answer that contradicts its own stated
            # information sufficiency, which the schema catches deliberately.
            raise LLMRejection(f"output failed schema validation: {exc}", raw) from exc

        return thesis, raw, latency_ms
