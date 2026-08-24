"""Kalshi request signing.

Kalshi authenticates every request with three headers:

* ``KALSHI-ACCESS-KEY``       -- the API key id
* ``KALSHI-ACCESS-TIMESTAMP`` -- current time in **milliseconds**
* ``KALSHI-ACCESS-SIGNATURE`` -- base64 RSA-PSS/SHA256 over
  ``timestamp + METHOD + path``

Two details cause most integration failures and are handled explicitly here:

1. The signed path is the **full path from the API root** including the
   ``/trade-api/v2`` prefix, and **without the query string**. So a request to
   ``https://external-api.kalshi.com/trade-api/v2/portfolio/orders?limit=5``
   signs ``/trade-api/v2/portfolio/orders``.
2. The timestamp is milliseconds, not seconds.

The private key is loaded once and kept in memory. It is never logged, never
placed in a config value, and only ever read from a file outside the repository.
"""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kat.common.errors import ConfigError
from kat.config.settings import KalshiCredentials

__all__ = ["KalshiSigner", "SignedHeaders", "signing_path"]


@dataclass(frozen=True, slots=True)
class SignedHeaders:
    key_id: str
    timestamp_ms: str
    signature: str

    def as_dict(self) -> dict[str, str]:
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": self.timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self.signature,
        }


def signing_path(url_or_path: str) -> str:
    """Return the exact path string Kalshi expects to be signed.

    Accepts either a full URL or a bare path, and strips the query string in
    both cases. Anything that is not an absolute path is rejected rather than
    guessed at, because a silently wrong signing path produces a 401 that looks
    like a credential problem and wastes a great deal of debugging time.
    """
    parts = urlsplit(url_or_path)
    path = parts.path if parts.scheme or parts.netloc else url_or_path.split("?", 1)[0]
    if not path.startswith("/"):
        raise ConfigError(f"signing path must be absolute, got {url_or_path!r}")
    return path


class KalshiSigner:
    """Holds one API identity and produces signed headers for it.

    Thread-safe: the underlying ``cryptography`` key object is immutable and
    ``sign`` is re-entrant, but the timestamp counter is guarded so two calls in
    the same millisecond cannot produce the same header set out of order.
    """

    def __init__(self, key_id: str, private_key: rsa.RSAPrivateKey, *, label: str = "kalshi") -> None:
        if not key_id:
            raise ConfigError(f"{label}: API key id is empty")
        self._key_id = key_id
        self._private_key = private_key
        self._label = label
        self._lock = threading.Lock()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_credentials(cls, creds: KalshiCredentials, *, label: str = "kalshi") -> KalshiSigner:
        if not creds.configured:
            raise ConfigError(
                f"{label}: credentials are not configured "
                "(set the api_key_id and private_key_path environment variables)"
            )
        return cls.from_key_file(creds.api_key_id, creds.resolve_key_path(), label=label)

    @classmethod
    def from_key_file(cls, key_id: str, path: Path, *, label: str = "kalshi") -> KalshiSigner:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ConfigError(f"{label}: cannot read private key at {path}: {exc}") from exc
        return cls.from_pem(key_id, data, label=label)

    @classmethod
    def from_pem(cls, key_id: str, pem: bytes, *, label: str = "kalshi") -> KalshiSigner:
        try:
            key = serialization.load_pem_private_key(pem, password=None)
        except Exception as exc:
            raise ConfigError(
                f"{label}: private key is not a readable unencrypted PEM key "
                f"({type(exc).__name__})"
            ) from exc
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ConfigError(
                f"{label}: Kalshi requires an RSA private key, got {type(key).__name__}"
            )
        return cls(key_id, key, label=label)

    # -- signing -------------------------------------------------------------

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, method: str, url_or_path: str, *, timestamp_ms: int | None = None) -> SignedHeaders:
        """Produce the three auth headers for one request."""
        path = signing_path(url_or_path)
        with self._lock:
            ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        message = f"{ts}{method.upper()}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return SignedHeaders(
            key_id=self._key_id,
            timestamp_ms=str(ts),
            signature=base64.b64encode(signature).decode("ascii"),
        )

    def headers(self, method: str, url_or_path: str) -> dict[str, str]:
        return self.sign(method, url_or_path).as_dict()

    def __repr__(self) -> str:  # never render the key material
        masked = f"{self._key_id[:4]}...{self._key_id[-4:]}" if len(self._key_id) > 8 else "****"
        return f"KalshiSigner(label={self._label!r}, key_id={masked!r})"
