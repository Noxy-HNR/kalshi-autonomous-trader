"""The generated service definitions must actually keep the platform running.

These assert the *settings that bite*, not the shape of the file. A scheduled
task that parses but stops the process after three days, or on battery, or when
the machine stops being idle, looks installed and is not supervision -- and the
failure it produces is indistinguishable from a crash.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from kat.engine.service import (
    systemd_unit,
    windows_task_command,
    windows_task_xml,
    write_service_files,
)

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def _settings(xml: str) -> ET.Element:
    root = ET.fromstring(xml)
    node = root.find("t:Settings", NS)
    assert node is not None
    return node


def _text(settings: ET.Element, path: str) -> str | None:
    node = settings.find(path, NS)
    return None if node is None else (node.text or "").strip()


def test_task_xml_is_well_formed() -> None:
    ET.fromstring(windows_task_xml())


def test_task_restarts_on_failure() -> None:
    """Without this the task starts the process once and never again."""
    settings = _settings(windows_task_xml())
    restart = settings.find("t:RestartOnFailure", NS)
    assert restart is not None, "no RestartOnFailure: a crash would stay down"
    assert _text(restart, "t:Interval") == "PT1M"
    assert int(_text(restart, "t:Count") or 0) >= 10


def test_task_has_no_execution_time_limit() -> None:
    """The default is three days, after which the task engine kills the process."""
    assert _text(_settings(windows_task_xml()), "t:ExecutionTimeLimit") == "PT0S"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("t:StopIfGoingOnBatteries", "false"),
        ("t:DisallowStartIfOnBatteries", "false"),
        ("t:IdleSettings/t:StopOnIdleEnd", "false"),
        ("t:RunOnlyIfIdle", "false"),
    ],
)
def test_task_survives_hostile_defaults(path: str, expected: str) -> None:
    """Each of these defaults to a value that stops a long-running process."""
    assert _text(_settings(windows_task_xml()), path) == expected


def test_task_runs_unelevated() -> None:
    root = ET.fromstring(windows_task_xml())
    principal = root.find("t:Principals/t:Principal", NS)
    assert principal is not None
    assert (principal.find("t:RunLevel", NS).text or "").strip() == "LeastPrivilege"


def test_task_runs_the_entrypoint_from_the_venv() -> None:
    root = ET.fromstring(windows_task_xml())
    exec_node = root.find("t:Actions/t:Exec", NS)
    assert exec_node is not None
    assert (exec_node.find("t:Arguments", NS).text or "").strip() == "-m kat.engine.entrypoint"
    assert (exec_node.find("t:Command", NS).text or "").strip().endswith("python.exe")


def test_only_one_instance_may_run() -> None:
    """Two engines against one account would each size against the full balance."""
    assert _text(_settings(windows_task_xml()), "t:MultipleInstancesPolicy") == "IgnoreNew"


def test_install_command_registers_from_the_xml() -> None:
    command = windows_task_command(Path("C:/x/kat-task.xml"))
    assert "/XML" in command
    assert "kat-task.xml" in command


def test_systemd_unit_restarts_and_allows_a_graceful_stop() -> None:
    unit = systemd_unit()
    assert "Restart=always" in unit
    # Shorter than the drain would make every restart look like a crash.
    assert "TimeoutStopSec=45" in unit


def test_write_service_files_emits_a_utf16_task(tmp_path: Path) -> None:
    """The task scheduler rejects UTF-8 XML with a message that never says so."""
    written = write_service_files(tmp_path)
    raw = written["task"].read_bytes()
    assert raw[:2] in (b"\xff\xfe", b"\xfe\xff"), "task XML must be UTF-16 with a BOM"
    ET.fromstring(raw.decode("utf-16"))


def test_generated_readme_does_not_promise_more_than_it_installs(tmp_path: Path) -> None:
    readme = write_service_files(tmp_path)["readme"].read_text(encoding="utf-8")
    assert "restarts on failure" in readme.lower()
    # It must still be honest that nothing watches the process beyond restarting it.
    assert "supervises it beyond" in readme


# -- the weekly retrain ------------------------------------------------------
#
# A scheduled job that can change what the platform trades is a different kind
# of thing from one that only reports. These pin it to the reporting kind.


def test_retrain_task_never_deploys() -> None:
    """The safety property of the whole feature.

    scripts/train_model.py registers a winner as a candidate and deploys only
    when asked, so the absence of --deploy in the task's arguments is what keeps
    an unattended run from changing what the platform trades.
    """
    from kat.engine.service import retrain_task_xml

    xml = retrain_task_xml()
    assert "--deploy" not in xml
    root = ET.fromstring(xml)
    arguments = root.find("t:Actions/t:Exec/t:Arguments", NS)
    assert arguments is not None
    assert (arguments.text or "").strip() == "-m scripts.train_model"


def test_retrain_task_runs_weekly_not_at_logon() -> None:
    from kat.engine.service import retrain_task_xml

    root = ET.fromstring(retrain_task_xml())
    triggers = root.find("t:Triggers", NS)
    assert triggers is not None
    kinds = [child.tag.split("}")[-1] for child in triggers]
    assert kinds == ["CalendarTrigger"]
    assert root.find("t:Triggers/t:CalendarTrigger/t:ScheduleByWeek", NS) is not None


def test_retrain_task_is_not_retried_on_failure() -> None:
    """The engine dying means positions are unwatched; a retrain dying means a
    report is late. Only one of those deserves 999 retries."""
    from kat.engine.service import retrain_task_xml

    settings = ET.fromstring(retrain_task_xml()).find("t:Settings", NS)
    assert settings is not None
    assert settings.find("t:RestartOnFailure", NS) is None


def test_retrain_task_has_its_own_name() -> None:
    """Sharing a name with the engine task would replace the engine's."""
    from kat.engine.service import RETRAIN_TASK_NAME, TASK_NAME, retrain_task_xml

    assert RETRAIN_TASK_NAME != TASK_NAME
    uri = ET.fromstring(retrain_task_xml()).find("t:RegistrationInfo/t:URI", NS)
    assert uri is not None
    assert uri.text == "\\" + RETRAIN_TASK_NAME


def test_the_engine_task_is_unchanged_by_parameterisation() -> None:
    """Adding the retrain must not have altered the daemon's own definition."""
    from kat.engine.service import windows_task_xml

    root = ET.fromstring(windows_task_xml())
    triggers = root.find("t:Triggers", NS)
    assert [c.tag.split("}")[-1] for c in triggers] == ["LogonTrigger"]
    assert root.find("t:Settings/t:RestartOnFailure", NS) is not None
    arguments = root.find("t:Actions/t:Exec/t:Arguments", NS)
    assert (arguments.text or "").strip() == "-m kat.engine.entrypoint"


def test_write_service_files_emits_the_retrain_task(tmp_path: Path) -> None:
    written = write_service_files(tmp_path)
    assert "retrain" in written
    raw = written["retrain"].read_bytes()
    assert raw[:2] in (b"\xff\xfe", b"\xfe\xff"), "task XML must be UTF-16"
    ET.fromstring(raw.decode("utf-16"))
    readme = written["readme"].read_text(encoding="utf-8")
    assert "never deploys" in readme.lower()
