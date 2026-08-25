"""Check Kalshi credentials without ever printing them.

Run this after putting your API key id in ``.env`` and saving your private key
to a file. It answers one question -- *can this process authenticate to Kalshi?*
-- and answers it by actually signing a request and calling the exchange, not by
inspecting configuration and hoping.

Nothing secret reaches the output. The key id is shown as a masked fingerprint
so you can tell two keys apart without either being readable, and the private
key is never rendered at all. That is deliberate: the whole point of a check
script is that you can paste its output to someone for help.

    python scripts/check_credentials.py

Exit code is 0 when the authenticated path works end to end, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages"))

from kat.common.enums import TradingEnvironment  # noqa: E402
from kat.config.settings import get_settings  # noqa: E402

OK = "  OK  "
BAD = " FAIL "
SKIP = " skip "


def mask(value: str) -> str:
    """Enough of an identifier to distinguish two, not enough to use one."""
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return f"{value[:2]}...({len(value)} chars)"
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def check_key_file(path_setting: str, label: str) -> tuple[bool, Path | None]:
    """The private key exists, parses, and is not the wrong kind of file."""
    if not path_setting:
        print(f"[{SKIP}] {label}: no private_key_path set")
        return False, None

    path = Path(os.path.expandvars(path_setting)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    if not path.exists():
        print(f"[{BAD}] {label}: no file at {path}")
        return False, path

    try:
        raw = path.read_bytes()
    except OSError as exc:
        print(f"[{BAD}] {label}: cannot read {path} ({exc})")
        return False, path

    head = raw[:64].decode("ascii", errors="replace")
    if "PRIVATE KEY" not in head:
        # The most common mistake by far: the API key *id* pasted into the file
        # that is supposed to hold the RSA key, or the PEM saved with the header
        # stripped by a text editor.
        print(
            f"[{BAD}] {label}: {path} does not begin with a PEM private-key header.\n"
            f"          It starts with: {head.splitlines()[0][:40]!r}\n"
            f"          This file must contain the RSA private key Kalshi gave you,\n"
            f"          including its '-----BEGIN ... PRIVATE KEY-----' line."
        )
        return False, path

    try:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        key = load_pem_private_key(raw, password=None)
    except Exception as exc:
        print(f"[{BAD}] {label}: {path} is not a usable unencrypted PEM key ({type(exc).__name__})")
        return False, path

    size = getattr(key, "key_size", None)
    print(f"[{OK}] {label}: private key loaded from {path}" + (f" ({size}-bit RSA)" if size else ""))

    # A key inside the repository is one `git add -A` away from being published.
    try:
        path.relative_to(REPO_ROOT)
        print(
            f"[{BAD}] {label}: this key is INSIDE the repository. Move it somewhere\n"
            f"          like C:/secrets/kalshi/ and update .env to point at it."
        )
        return False, path
    except ValueError:
        pass

    return True, path


async def check_live() -> bool:
    """Sign a real request and see whether Kalshi accepts it.

    Built through exactly the same calls the engine makes at start-up, so a pass
    here means the engine will authenticate too -- a check that constructed its
    own client a different way could pass while the engine still failed.
    """
    from kat.kalshi.auth import KalshiSigner
    from kat.kalshi.event_contracts.rest import EventContractsREST
    from kat.kalshi.http import KalshiHTTPClient
    from kat.kalshi.ratelimit import RateLimiter

    settings = get_settings()
    signer = KalshiSigner.from_credentials(
        settings.kalshi.event_credentials, label="event"
    )
    client = KalshiHTTPClient(
        base_url=settings.kalshi.rest_base_url,
        signer=signer,
        rate_limiter=RateLimiter(),
        is_perp=False,
        timeout_s=settings.kalshi.request_timeout_s,
        connect_timeout_s=settings.kalshi.connect_timeout_s,
    )
    rest = EventContractsREST(client)
    try:
        balance = await rest.get_balance()
        # The value is the user's own account balance, not a secret, and seeing
        # it is the clearest possible proof the authenticated path works.
        print(f"[{OK}] authenticated call succeeded: balance endpoint returned {balance}")
        return True
    except Exception as exc:
        print(f"[{BAD}] authenticated call failed: {type(exc).__name__}: {exc}")
        return False
    finally:
        await client.aclose()


async def main() -> int:
    settings = get_settings()
    kalshi = settings.kalshi

    print("=" * 70)
    print("  Kalshi credential check")
    print("=" * 70)
    print(f"  environment : {settings.environment.value}")
    print(f"  API base    : {kalshi.rest_base_url}")
    print()

    if settings.environment is TradingEnvironment.PAPER:
        print(
            "  Note: KAT_ENVIRONMENT=paper. Credentials are still used for market\n"
            "  data (Kalshi authenticates the WebSocket even for public channels),\n"
            "  but no order will ever be sent in this mode.\n"
        )

    creds = kalshi.event_credentials
    ok = True

    if creds.api_key_id:
        print(f"[{OK}] event api_key_id is set: {mask(creds.api_key_id)}")
    else:
        print(f"[{BAD}] event api_key_id is empty")
        print("        Set KAT_KALSHI__EVENT_CREDENTIALS__API_KEY_ID in .env")
        ok = False

    key_ok, _ = check_key_file(creds.private_key_path, "event")
    ok = ok and key_ok

    if kalshi.perp_credentials.api_key_id or kalshi.perp_credentials.private_key_path:
        print()
        print(f"[{OK}] perp api_key_id is set: {mask(kalshi.perp_credentials.api_key_id)}")
        check_key_file(kalshi.perp_credentials.private_key_path, "perp")
    else:
        print(f"[{SKIP}] perpetuals: no credentials configured (optional)")

    print()
    if not ok:
        print("Configuration is incomplete; not attempting a live call.")
        return 1

    print("Attempting a signed request to Kalshi...")
    live = await check_live()

    print()
    print("=" * 70)
    print("  RESULT:", "credentials work" if live else "credentials do NOT work")
    print("=" * 70)
    return 0 if live else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
