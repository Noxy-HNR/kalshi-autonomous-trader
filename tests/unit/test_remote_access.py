"""Remote access must reach your devices and nothing else.

The dashboard API has no authentication -- it can engage kill switches, change
the operating mode and move risk limits -- so the network is the authentication.
These tests pin the refusals, because the failure they prevent is someone
reaching for remote access, binding every interface, and publishing that API to
whatever network the machine happens to be on.
"""

from __future__ import annotations

import pytest

from kat.net.tailscale import (
    TAILSCALE_NETWORK,
    RemoteAccessError,
    is_loopback,
    is_tailscale_address,
    remote_access_status,
    resolve_bind_host,
)

TAILNET = "100.101.102.103"


# -- what counts as a tailnet address ---------------------------------------


def test_tailscale_range_is_the_rfc6598_shared_space() -> None:
    """100.64.0.0/10, which is not routable on the public internet."""
    assert str(TAILSCALE_NETWORK) == "100.64.0.0/10"


@pytest.mark.parametrize("host", ["100.64.0.1", "100.101.102.103", "100.127.255.254"])
def test_addresses_inside_the_range_are_recognised(host: str) -> None:
    assert is_tailscale_address(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "100.63.255.255",  # just below the range
        "100.128.0.0",  # just above it
        "100.200.1.5",  # public 100.x space, not a tailnet
        "192.168.1.50",  # ordinary LAN
        "10.0.0.5",  # private, but not a tailnet
        "8.8.8.8",
        "not-an-address",
        "",
    ],
)
def test_addresses_outside_the_range_are_rejected(host: str) -> None:
    """100.x is not automatically Tailscale; only 100.64/10 is."""
    assert is_tailscale_address(host) is False


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.53", "::1", "localhost"])
def test_loopback_is_recognised(host: str) -> None:
    assert is_loopback(host) is True


# -- the refusals ------------------------------------------------------------


@pytest.mark.parametrize("host", ["0.0.0.0", "::", ""])
def test_binding_every_interface_is_refused(host: str) -> None:
    """There is no configuration that allows this, remote access on or off."""
    for mode in ("off", "tailscale"):
        with pytest.raises(RemoteAccessError, match="every network interface"):
            resolve_bind_host(host, remote_access=mode)


@pytest.mark.parametrize("host", ["192.168.1.50", "10.0.0.5", "172.16.4.2"])
def test_binding_a_lan_address_is_refused(host: str) -> None:
    """A LAN is not private by construction: anyone on the wifi is on it."""
    with pytest.raises(RemoteAccessError, match="neither a loopback nor a Tailscale"):
        resolve_bind_host(host, remote_access="off")


def test_the_refusal_says_what_to_do_instead() -> None:
    with pytest.raises(RemoteAccessError) as exc:
        resolve_bind_host("0.0.0.0", remote_access="off")
    assert "KAT_API_REMOTE_ACCESS=tailscale" in str(exc.value)


# -- what it resolves to -----------------------------------------------------


def test_loopback_is_kept_when_remote_access_is_off() -> None:
    assert resolve_bind_host("127.0.0.1", remote_access="off") == "127.0.0.1"


def test_an_explicit_tailnet_address_is_allowed_without_the_mode() -> None:
    """Someone who already knows their address may just set it."""
    assert resolve_bind_host(TAILNET, remote_access="off") == TAILNET


def test_tailscale_mode_binds_the_tailnet_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kat.net.tailscale.find_tailscale_address", lambda: TAILNET)
    assert resolve_bind_host("127.0.0.1", remote_access="tailscale") == TAILNET


def test_tailscale_mode_falls_back_to_loopback_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Down the *safe* direction.

    A trading platform that will not start because a VPN is down is worse than
    one that starts reachable only from its own machine. The fallback reduces
    exposure rather than increasing it, and it is a warning in the log.
    """
    monkeypatch.setattr("kat.net.tailscale.find_tailscale_address", lambda: None)
    assert resolve_bind_host("127.0.0.1", remote_access="tailscale") == "127.0.0.1"


def test_tailscale_mode_never_falls_back_to_a_wider_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kat.net.tailscale.find_tailscale_address", lambda: None)
    resolved = resolve_bind_host("127.0.0.1", remote_access="tailscale")
    assert resolved not in ("0.0.0.0", "::")
    assert is_loopback(resolved)


def test_a_bogus_tailscale_answer_is_not_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    """If detection ever returned a non-tailnet address, binding it would be
    exactly the exposure this module exists to prevent."""
    monkeypatch.setattr("kat.net.tailscale.find_tailscale_address", lambda: "0.0.0.0")
    resolved = resolve_bind_host("127.0.0.1", remote_access="tailscale")
    assert resolved != "0.0.0.0"


# -- what the dashboard is told ---------------------------------------------


def test_status_reports_both_installed_and_connected() -> None:
    """Installed-but-logged-out is a different problem from not installed, and
    the operator needs to be told which one they have."""
    status = remote_access_status()
    assert set(status) >= {"installed", "connected", "address", "env_override"}
    assert isinstance(status["installed"], bool)
    assert isinstance(status["connected"], bool)


def test_status_address_is_absent_when_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kat.net.tailscale.find_tailscale_address", lambda: None)
    status = remote_access_status()
    assert status["connected"] is False
    assert status["address"] is None


def test_status_reports_the_address_when_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kat.net.tailscale.find_tailscale_address", lambda: TAILNET)
    status = remote_access_status()
    assert status["connected"] is True
    assert status["address"] == TAILNET


# -- the setting -------------------------------------------------------------


def test_settings_offer_no_option_that_binds_everything() -> None:
    """The type itself is the guarantee; there is no "all" to select."""
    from typing import get_args

    from kat.config.settings import AppSettings

    allowed = get_args(AppSettings.model_fields["api_remote_access"].annotation)
    assert set(allowed) == {"off", "tailscale"}


def test_remote_access_defaults_to_off() -> None:
    """The field's default, not an instance's value.

    ``AppSettings()`` reads the developer's own ``.env``, so asserting on an
    instance would make this test pass or fail depending on whose machine it
    runs on.
    """
    from kat.config.settings import AppSettings

    assert AppSettings.model_fields["api_remote_access"].default == "off"


def test_status_reads_the_configured_mode_not_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pydantic-settings loads .env without exporting it.

    Reading os.environ here reported "off" for a machine whose .env said
    "tailscale" -- the one reading that makes a working configuration look like
    an unset one, and sends the operator to set a variable they already set.
    """
    monkeypatch.delenv("KAT_API_REMOTE_ACCESS", raising=False)
    assert remote_access_status(requested="tailscale")["env_override"] == "tailscale"
    assert remote_access_status(requested="tailscale")["requested"] == "tailscale"


def test_status_rejects_a_non_tailnet_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kat.net.tailscale.find_tailscale_address", lambda: "192.168.1.9")
    status = remote_access_status(requested="tailscale")
    assert status["connected"] is False
    assert status["address"] is None
