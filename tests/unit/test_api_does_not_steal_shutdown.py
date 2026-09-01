"""The dashboard API must not take the platform's shutdown signals.

The platform's SIGINT/SIGTERM handler flushes the async writer and records a
clean shutdown. Uvicorn's handler stops uvicorn. If uvicorn's wins, the process
dies with the writer's queue unflushed and the next start reports an unclean
shutdown it did not have.

The entrypoint used to prevent that with ``install_signal_handlers = False``.
Uvicorn removed that attribute in 0.30, so the assignment quietly created an
unused one and ``serve()`` went on installing its handlers regardless -- the
kind of failure that leaves a comment claiming a problem is solved and no way to
notice it is not. These tests check the mechanism against the installed uvicorn
rather than trusting the comment.
"""

from __future__ import annotations

import contextlib
import signal

import pytest
import uvicorn
from uvicorn.config import Config


async def _app(scope, receive, send):  # pragma: no cover - never invoked
    return None


def _server() -> uvicorn.Server:
    return uvicorn.Server(Config(app=_app, host="127.0.0.1", port=9, log_level="warning"))


@pytest.fixture
def restore_sigint():
    original = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, original)


class TestUvicornSignalCapture:
    def test_the_attribute_the_entrypoint_neutralises_still_exists(self):
        """A canary on uvicorn's API.

        The entrypoint replaces ``capture_signals``. If a future uvicorn renames
        it, the replacement silently stops working and the platform goes back to
        losing its shutdown handler -- exactly the regression this file exists
        for. Failing here is the cheapest possible way to find that out.
        """
        assert hasattr(_server(), "capture_signals")

    def test_the_removed_attribute_is_confirmed_gone(self):
        """Guards against reintroducing the no-op that started this.

        If this ever fails, uvicorn has brought the option back and the
        entrypoint should use it instead of patching a context manager.
        """
        assert not hasattr(uvicorn.Server, "install_signal_handlers")

    def test_uvicorn_would_otherwise_replace_the_platforms_handler(self, restore_sigint):
        """The behaviour being defended against, demonstrated."""
        def platform_handler(signum, frame):  # pragma: no cover - never invoked
            return None

        signal.signal(signal.SIGINT, platform_handler)
        server = _server()

        with server.capture_signals():
            assert signal.getsignal(signal.SIGINT) is not platform_handler

    def test_the_neutralised_server_leaves_the_handler_alone(self, restore_sigint):
        """The fix, applied exactly as the entrypoint applies it."""
        def platform_handler(signum, frame):  # pragma: no cover - never invoked
            return None

        signal.signal(signal.SIGINT, platform_handler)
        server = _server()
        server.capture_signals = contextlib.nullcontext  # type: ignore[assignment]

        with server.capture_signals():
            assert signal.getsignal(signal.SIGINT) is platform_handler
        assert signal.getsignal(signal.SIGINT) is platform_handler

    def test_the_entrypoint_neutralises_it(self):
        """The source actually does what the tests above validate."""
        from pathlib import Path

        import kat.engine.entrypoint as entrypoint

        source = Path(entrypoint.__file__).read_text(encoding="utf-8")
        assert "capture_signals = contextlib.nullcontext" in source
        # The comment above the fix names the old attribute deliberately, so
        # look only at code: assigning it is the bug, explaining it is not.
        code = [
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        ]
        assert not [line for line in code if "install_signal_handlers" in line]
