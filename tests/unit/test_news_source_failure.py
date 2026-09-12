"""What a failed news source says about itself.

``news.source_failed`` logged ``"error": ""`` 1,841 times, for ``httpx.ReadError``,
whose ``str()`` is empty: a line that reports a feed failed and withholds every
fact about how. A read error, a refused connection and a TLS failure are
different problems, and for as long as this lasted they were the same line.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kat.common.timeutil import SimulatedClock
from kat.config.settings import NewsSettings
from kat.news import pipeline as pipeline_module
from kat.news.pipeline import NewsPipeline


class _FailingSource:
    source_id = "wire_a"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def fetch(self, client: httpx.AsyncClient) -> list[Any]:
        raise self._exc


class _RecordingLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        def record(event: str, **fields: Any) -> None:
            self.records.append((level, event, fields))

        return record


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (httpx.ReadError(""), "ReadError"),
        (httpx.ConnectError("connection refused"), "connection refused"),
    ],
    ids=["empty-message", "with-message"],
)
async def test_a_failed_source_names_the_exception(monkeypatch, exc, error):
    recorder = _RecordingLog()
    monkeypatch.setattr(pipeline_module, "log", recorder)
    pipeline = NewsPipeline(
        NewsSettings(), SimulatedClock(start_ms=2_000_000), sources=[_FailingSource(exc)]
    )
    try:
        result = await pipeline.poll()
    finally:
        await pipeline.aclose()

    failures = [
        (level, fields)
        for level, event, fields in recorder.records
        if event == "news.source_failed"
    ]
    assert failures == [
        (
            "warning",
            {"source": "wire_a", "error": error, "error_type": type(exc).__name__},
        )
    ]
    assert result.source_failures == [f"wire_a: {type(exc).__name__}"]
