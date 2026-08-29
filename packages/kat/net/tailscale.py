"""Reaching the dashboard from another machine, without exposing it.

The dashboard has no authentication and it never will: it can engage kill
switches, change the operating mode and move risk limits, and putting a login
form in front of that would mean inventing a session and password story for a
single-operator tool. The decision recorded in ``docs/SECURITY.md`` is that the
*network* is the authentication -- the API answers only on addresses that are
already private.

Loopback is the default and needs nothing. This module is the other supported
option: a Tailscale address.

Why Tailscale specifically
--------------------------
A Tailscale address is a WireGuard tunnel endpoint. Binding to one is not the
same as binding to a LAN address and not remotely the same as binding to
``0.0.0.0``:

* the address only exists while the tunnel is up, and only devices on your
  tailnet can route to it -- there is no path from the public internet;
* traffic is end-to-end encrypted by WireGuard, so the absence of TLS on the
  dashboard is not the exposure it would otherwise be;
* access is revoked by removing the device from the tailnet, which is a control
  that exists outside this process and cannot be undone by anything running in
  it.

What this module will not do
----------------------------
It refuses to hand back ``0.0.0.0`` or any address that is neither loopback nor
inside Tailscale's range. That refusal is the point: the failure it prevents is
someone reaching for remote access, binding to every interface, and publishing
an unauthenticated kill-switch API to whatever network the machine is on.

The Tailscale range is ``100.64.0.0/10``. That is the shared address space from
RFC 6598, which Tailscale uses for its tailnet addresses and which is not
routable on the public internet.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess
from pathlib import Path

from kat.observability.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "TAILSCALE_NETWORK",
    "RemoteAccessError",
    "find_tailscale_address",
    "is_tailscale_address",
    "remote_access_status",
    "resolve_bind_host",
    "tailscale_cli",
]

#: RFC 6598 shared address space, which is what a tailnet hands out.
TAILSCALE_NETWORK = ipaddress.ip_network("100.64.0.0/10")

#: Where the Windows installer puts the CLI. Checked before PATH because a
#: service started at logon does not reliably inherit a user's PATH.
_WINDOWS_CLI_PATHS = (
    r"C:\Program Files\Tailscale\tailscale.exe",
    r"C:\Program Files (x86)\Tailscale\tailscale.exe",
)

_UNIX_CLI_PATHS = (
    "/usr/bin/tailscale",
    "/usr/local/bin/tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
)


class RemoteAccessError(RuntimeError):
    """Remote access was requested and cannot be provided safely."""


def is_tailscale_address(host: str) -> bool:
    """Whether ``host`` is an address inside the tailnet range."""
    try:
        return ipaddress.ip_address(host) in TAILSCALE_NETWORK
    except ValueError:
        return False


def is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def tailscale_cli() -> str | None:
    """Path to the Tailscale CLI, or None if it is not installed."""
    for candidate in (*_WINDOWS_CLI_PATHS, *_UNIX_CLI_PATHS):
        if Path(candidate).exists():
            return candidate
    return shutil.which("tailscale")


def _address_from_cli() -> str | None:
    """Ask Tailscale for this machine's tailnet address.

    Preferred over inspecting interfaces because it is the authoritative answer
    and it distinguishes "installed but logged out" from "not installed" -- both
    of which look identical from the interface list.
    """
    cli = tailscale_cli()
    if cli is None:
        return None
    try:
        result = subprocess.run(
            [cli, "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("tailscale.cli_failed", error=str(exc))
        return None
    if result.returncode != 0:
        # Logged out, or the daemon is not running. The message is the useful
        # part; it says which of those it is.
        log.debug("tailscale.cli_error", detail=(result.stderr or "").strip()[:200])
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if is_tailscale_address(candidate):
            return candidate
    return None


def _address_from_interfaces() -> str | None:
    """Find a tailnet address among the machine's own addresses.

    A fallback for when the CLI is not on this account's PATH. It cannot tell a
    stale address from a live one, so it is only consulted second.
    """
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
    except OSError:
        return None
    for info in infos:
        candidate = info[4][0]
        if is_tailscale_address(candidate):
            return candidate
    return None


def find_tailscale_address() -> str | None:
    """This machine's tailnet address, or None if it does not have one."""
    return _address_from_cli() or _address_from_interfaces()


def resolve_bind_host(configured: str, *, remote_access: str) -> str:
    """The address the API should bind to, or an explanation of the refusal.

    ``remote_access="off"`` keeps whatever is configured, after checking it is
    private. ``"tailscale"`` looks up the tailnet address and uses it.

    Falling back is deliberately toward *less* exposure. If remote access was
    asked for and Tailscale cannot supply an address -- not installed, logged
    out, daemon down -- this returns loopback with a warning rather than
    refusing to start. A trading platform that will not come up because a VPN is
    down is worse than one that comes up reachable only from its own machine,
    and the failure is loud in the log either way.

    Binding to every interface is refused outright. There is no configuration
    that allows it, because an unauthenticated API that can engage a kill switch
    and move risk limits does not get published to an unknown network by
    accident.
    """
    if configured in ("0.0.0.0", "::", ""):
        raise RemoteAccessError(
            f"KAT_API_HOST={configured!r} would publish the dashboard API on every "
            "network interface. The API has no authentication -- it can engage kill "
            "switches and change risk limits -- so this is refused. Use the default "
            "127.0.0.1, or set KAT_API_REMOTE_ACCESS=tailscale to reach it from "
            "another device over your tailnet."
        )

    if remote_access == "tailscale":
        address = find_tailscale_address()
        # Re-checked rather than trusted. Detection parses output from another
        # program, and the one thing this module must never do is hand back an
        # address that is not inside the tailnet -- so the answer is validated
        # here too, where the binding decision is actually made.
        if address is not None and not is_tailscale_address(address):
            log.error(
                "remote_access.detected_address_rejected",
                host=address,
                note="not inside 100.64.0.0/10; falling back to loopback",
            )
            address = None
        if address is not None:
            log.info("remote_access.tailscale", host=address)
            return address
        log.warning(
            "remote_access.tailscale_unavailable",
            note=(
                "no tailnet address found, so the API stays on loopback and is not "
                "reachable from other devices"
            ),
            cli_present=tailscale_cli() is not None,
            remedy="install Tailscale and run `tailscale up`, then restart",
        )
        return "127.0.0.1"

    if not (is_loopback(configured) or is_tailscale_address(configured)):
        raise RemoteAccessError(
            f"KAT_API_HOST={configured!r} is neither a loopback nor a Tailscale "
            "address. The dashboard API has no authentication, so it is only bound "
            "to networks that are private by construction. Use 127.0.0.1, or set "
            "KAT_API_REMOTE_ACCESS=tailscale."
        )
    return configured


def remote_access_status(requested: str | None = None) -> dict[str, object]:
    """What the dashboard shows about remote access.

    ``requested`` is the configured mode. It is read from settings rather than
    from the environment because pydantic-settings loads ``.env`` without
    exporting it: reading ``os.environ`` here reported "off" for a machine whose
    ``.env`` said ``tailscale``, which is the one reading that makes a working
    configuration look like an unset one.
    """
    if requested is None:
        try:
            from kat.config.settings import get_settings

            requested = get_settings().api_remote_access
        except Exception:  # settings unavailable in some tooling contexts
            requested = os.environ.get("KAT_API_REMOTE_ACCESS", "off")

    address = find_tailscale_address()
    if address is not None and not is_tailscale_address(address):
        address = None
    return {
        "installed": tailscale_cli() is not None,
        "connected": address is not None,
        # The address is private to the tailnet, and the operator needs to see
        # it to type it into another device.
        "address": address,
        "requested": requested,
        # Kept under the old name too: the dashboard reads this field.
        "env_override": requested,
    }
