"""Running as a long-lived service on Windows and Linux.

The platform is a native process. This module is about *keeping* it running
across reboots without introducing a shell script, a batch file, or a
supervisor process into the trading path -- all three are explicitly out of
scope, and all three would put a piece of the system's reliability somewhere the
system cannot see it.

What is actually needed differs by platform, and neither needs anything at
runtime:

**Windows.** A Scheduled Task set to run at logon (or at startup, for a service
account) with "restart on failure". Registered once with ``schtasks``; after
that the operating system starts the process and nothing else is involved. A
true Windows Service would need pywin32 and a service wrapper, which buys
nothing here -- the application already handles its own start-up, shutdown,
health and recovery.

**Linux.** A systemd unit, which is the same idea with better ergonomics.

Both are *generated* here rather than executed. Installing a startup entry
changes the machine, so this module prints what it would install and writes the
unit file; running it is the operator's deliberate action.

Safe startup
------------
Neither entry arms live trading. The process reads its configuration at start,
and live trading needs ``KAT_ENVIRONMENT=live`` *and*
``KAT_LIVE_TRADING_ARMED=true`` to be set in that configuration. A machine that
reboots comes back in whatever mode its configuration says -- which, unless
someone deliberately changed it, is paper.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from kat.config.settings import REPO_ROOT, get_settings

__all__ = [
    "systemd_unit",
    "windows_task_command",
    "windows_task_xml",
    "write_service_files",
]

TASK_NAME = "KalshiAutonomousTrader"


def _python_executable() -> str:
    """The interpreter that should run the service.

    Prefers the virtualenv's own executable so the task does not depend on PATH
    resolution at logon time, which is a common way for a scheduled task to work
    when tested by hand and fail on reboot.
    """
    venv = REPO_ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    candidate = venv / ("python.exe" if sys.platform == "win32" else "python")
    if candidate.exists():
        return str(candidate)
    return sys.executable


def windows_task_xml(user: str | None = None) -> str:
    """A Scheduled Task definition that survives the ways Windows kills daemons.

    Registered from XML rather than from ``schtasks`` switches because the four
    settings that matter here cannot be expressed on the command line, and all
    four of their defaults are wrong for a process that is supposed to run for
    weeks:

    ``RestartOnFailure``
        Not expressible via ``schtasks /Create`` switches at all. Without it the
        task starts the process once at logon and, if that process ever dies,
        does nothing -- which is how a five-hour run ended with the platform
        down and nothing bringing it back.

    ``ExecutionTimeLimit``
        Defaults to three days, after which the task engine stops the process.
        A trading platform that is killed every 72 hours by its own supervisor
        is worse than one that is not supervised, because the failure looks like
        a crash.

    ``StopIfGoingOnBatteries`` / ``DisallowStartIfOnBatteries``
        Both default to true. On a laptop that means the platform stops when the
        charger is unplugged and refuses to start on battery.

    ``StopOnIdleEnd``
        Defaults to true: the task is stopped when the machine stops being idle.

    ``LeastPrivilege`` matches the previous ``/RL LIMITED``: nothing here needs
    elevation, and a trading process running as Administrator is a larger blast
    radius for no benefit.
    """
    python = _python_executable()
    account = user or _current_user()
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Kalshi Autonomous Trader. Restarts on failure; never arms live trading by itself.</Description>
    <URI>\\{TASK_NAME}</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{account}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{account}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>false</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{python}</Command>
      <Arguments>-m kat.engine.entrypoint</Arguments>
      <WorkingDirectory>{REPO_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def windows_task_command(xml_path: Path | None = None) -> str:
    """The ``schtasks`` command that registers the task from its XML definition."""
    path = xml_path or (REPO_ROOT / "deploy" / "kat-task.xml")
    return f'schtasks /Create /TN "{TASK_NAME}" /XML "{path}" /F'


def _current_user() -> str:
    r"""The account the task should run as, in DOMAIN\user form where available."""
    domain = os.environ.get("USERDOMAIN")
    user = os.environ.get("USERNAME", "")
    return f"{domain}\\{user}" if domain and user else user


def windows_delete_command() -> str:
    return f'schtasks /Delete /TN "{TASK_NAME}" /F'


def systemd_unit(user: str = "kat", working_dir: Path | None = None) -> str:
    """A systemd unit for the Linux server migration.

    ``Restart=always`` with a delay handles a crash. ``TimeoutStopSec`` gives the
    application time to run its own graceful shutdown -- draining the write queue
    and recording a clean-shutdown marker -- rather than being killed mid-flush,
    which would make the next start treat it as a crash.
    """
    working = working_dir or REPO_ROOT
    python = f"{working}/.venv/bin/python"
    return f"""[Unit]
Description=Kalshi Autonomous Trader
Documentation=https://github.com/Noxy-HNR/kalshi-autonomous-trader
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={working}
Environment=PYTHONUNBUFFERED=1
# Secrets live outside the repository; this file is read by systemd, not by git.
EnvironmentFile=-/etc/kat/kat.env
ExecStart={python} -m kat.engine.entrypoint
Restart=always
RestartSec=10
# Long enough for the graceful shutdown path to drain the write queue and record
# a clean-shutdown marker. Killing it sooner makes the next start treat this as
# a crash and demand a full reconciliation before trading.
TimeoutStopSec=45
KillSignal=SIGTERM

# The platform needs no elevated privileges at runtime.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={working}/logs {working}/data

[Install]
WantedBy=multi-user.target
"""


def write_service_files(output_dir: Path | None = None) -> dict[str, Path]:
    """Write the unit file and a README describing installation.

    Writes files; installs nothing. Registering a startup entry modifies the
    machine and is the operator's call to make.
    """
    target = output_dir or (REPO_ROOT / "deploy")
    target.mkdir(parents=True, exist_ok=True)

    unit_path = target / "kat.service"
    unit_path.write_text(systemd_unit(), encoding="utf-8")

    # UTF-16 with a BOM: the task scheduler rejects the XML otherwise, and the
    # error it gives ("The task XML is malformed") does not mention encoding.
    task_path = target / "kat-task.xml"
    task_path.write_text(windows_task_xml(), encoding="utf-16")

    readme = target / "README.md"
    settings = get_settings()
    readme.write_text(
        f"""# Running the platform continuously

Generated by `kat.engine.service`. Nothing here has been installed; these are
the commands to run when you want the platform to survive a reboot.

The application handles its own start-up, shutdown, reconnection, health checks,
recovery and graceful termination. Neither entry below supervises it beyond
restarting the process if it exits -- but both do restart it, which is the part
that matters after an unattended crash.

## Windows (scheduled task, runs at logon, restarts on failure)

Registered from `kat-task.xml` rather than from `schtasks` switches, because
restart-on-failure cannot be expressed as a switch and three of the task
engine's defaults would otherwise stop a long-running process: it halts tasks
after three days, stops them when the machine goes on battery, and stops them
when the machine stops being idle. The XML overrides all four.

```
{windows_task_command(task_path)}
```

To remove it:

```
{windows_delete_command()}
```

Runs at LIMITED integrity: the platform needs no elevation, and running a
trading process as Administrator enlarges the blast radius for no benefit.

## Linux (systemd)

```
sudo cp {unit_path.name} /etc/systemd/system/kat.service
sudo systemctl daemon-reload
sudo systemctl enable --now kat
```

Secrets go in `/etc/kat/kat.env`, which is outside the repository and read by
systemd rather than committed.

## What a reboot does NOT do

It does not arm live trading. The process reads its configuration at start, and
real orders require **both** `KAT_ENVIRONMENT=live` and
`KAT_LIVE_TRADING_ARMED=true`. This machine is currently configured as:

- environment: **{settings.environment.value}**
- live trading armed: **{settings.live_trading_armed}**
- operating mode: **{settings.operating_mode.value}**

A restart comes back in exactly that state.

## Verifying it came back up

```
curl -s http://{settings.api_host}:{settings.api_port}/api/health
```

After any restart the platform reconciles its state against the exchange before
trading. If reconciliation cannot prove the two agree, it engages the risk kill
switch and stays in an observe-only mode until a human resolves it -- it never
resumes anyway.
""",
        encoding="utf-8",
    )
    return {"unit": unit_path, "task": task_path, "readme": readme}


def main() -> None:
    """Print what would be installed, and write the unit file."""
    paths = write_service_files()
    print(f"Wrote {paths['unit']}")
    print(f"Wrote {paths['readme']}")
    print()
    if sys.platform == "win32":
        print("To start at logon on this machine, run:")
        print(f"  {windows_task_command()}")
        if shutil.which("schtasks") is None:
            print("  (schtasks was not found on PATH)")
    else:
        print("To install the systemd unit:")
        print(f"  sudo cp {paths['unit']} /etc/systemd/system/kat.service")
        print("  sudo systemctl daemon-reload && sudo systemctl enable --now kat")
    print()
    print("Nothing has been installed. Run the command above deliberately.")


if __name__ == "__main__":
    main()
