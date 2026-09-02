from __future__ import annotations

import subprocess

from ageneval.task.sandbox import cleanup


def test_scoped_cleanup_filters_campaign_and_trial_without_legacy_sweep(monkeypatch) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], *, timeout: int | None = None):
        commands.append(command)
        stdout = "container-1\n" if "ps" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(cleanup, "_docker_available", lambda: True)
    monkeypatch.setattr(cleanup, "_run", run)
    monkeypatch.setattr(
        cleanup,
        "_image_orphan_ids",
        lambda _prefixes: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    removed = cleanup.sweep_sandbox_containers(
        campaign_id="campaign-1",
        trial_id="trial-1",
        include_image_orphans=True,
    )

    assert removed == ["container-1"]
    list_command = commands[0]
    assert "label=a2e.sandbox=1" in list_command
    assert "label=a2e.campaign_id=campaign-1" in list_command
    assert "label=a2e.trial_id=trial-1" in list_command
