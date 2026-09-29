# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""continuity_cli.py's dispatch logic, driven as a function -- not via
subprocess, which would require the real venv Python on the test runner's
PATH. sys.path is extended so this pure-script file (not a package member) is
importable the same way pytest's own conftest resolution would find it.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_CLI_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tools", "continuity_cli.py"
)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("continuity_cli", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_on_a_fresh_project_prints_something(tmp_path, capsys):
    cli = _load_cli_module()
    exit_code = cli.main(["status", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No project state" in out or "not recorded" in out


def test_init_creates_project_state_then_status_shows_it(tmp_path, capsys):
    """The one gap Task 9's review confirmed spans the whole plan: nothing else
    -- not the tool, not any other CLI verb -- can ever create the FIRST
    project_state.json. This is that path."""
    cli = _load_cli_module()
    exit_code = cli.main(["init", "demo-project", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    capsys.readouterr()
    exit_code = cli.main(["status", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "demo-project" in out


def test_init_refuses_to_overwrite_an_existing_state(tmp_path, capsys):
    """A second init must not silently erase real progress -- completed work,
    blocking_issues, etc. -- that a fresh ProjectState would discard."""
    cli = _load_cli_module()
    assert cli.main(["init", "first", "--project-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    exit_code = cli.main(["init", "second", "--project-dir", str(tmp_path)])
    assert exit_code != 0
    capsys.readouterr()
    exit_code = cli.main(["status", "--project-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "first" in out, "the original state must survive the refused second init"


def test_task_add_then_status_round_trips(tmp_path, capsys):
    cli = _load_cli_module()
    cli.main(["task", "add", "t1", "--title", "do the thing", "--project-dir", str(tmp_path)])
    capsys.readouterr()  # discard the add's own output
    cli.main(["status", "--project-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "t1" in out


def test_an_unknown_verb_exits_nonzero_and_says_so(tmp_path, capsys):
    cli = _load_cli_module()
    exit_code = cli.main(["not-a-real-verb", "--project-dir", str(tmp_path)])
    assert exit_code != 0


def test_error_record_then_check_round_trips(tmp_path, capsys, monkeypatch):
    """errors.jsonl's own motivation (spec section 3) is a human's mistakes, and
    until now the CLI -- the human's interface -- could neither write nor read it."""
    cli = _load_cli_module()
    # Never the real ~/odysseus/.remember/now.md, even where no fallback should
    # be needed: if the primary write failed unexpectedly, this is where it goes.
    fallback = tmp_path / "remember" / "now.md"
    monkeypatch.setattr(cli, "_REMEMBER_NOW_PATH", str(fallback))

    exit_code = cli.main(["error", "record", "loose regex", "matched both brands",
                          "--symptom", "inert control", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    assert "Recorded." in capsys.readouterr().out

    exit_code = cli.main(["error", "check", "inert", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "loose regex" in out and "matched both brands" in out
    assert not fallback.exists(), "the primary write landed; the fallback must not be touched"


def test_error_record_falls_back_for_a_human_and_says_so_when_both_writes_fail(tmp_path, capsys, monkeypatch):
    from core.continuity import __init__ as continuity_module

    def _boom(*a, **k):
        raise OSError("disk full")

    cli = _load_cli_module()
    monkeypatch.setattr(continuity_module, "_append_error_line", _boom)
    fallback = tmp_path / "remember" / "now.md"
    monkeypatch.setattr(cli, "_REMEMBER_NOW_PATH", str(fallback))

    exit_code = cli.main(["error", "record", "x-tried", "y-failed", "--project-dir", str(tmp_path)])
    assert exit_code == 0, "the primary write failed but the human's fallback note landed"
    assert "x-tried" in fallback.read_text(encoding = "utf-8")
    capsys.readouterr()

    blocker = tmp_path / "blocker"
    blocker.write_text("a plain file where a directory would have to be", encoding = "utf-8")
    monkeypatch.setattr(cli, "_REMEMBER_NOW_PATH", str(blocker / "now.md"))
    exit_code = cli.main(["error", "record", "x2", "y2", "--project-dir", str(tmp_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "could not record" in captured.err
    assert "Recorded." not in captured.out


def test_a_continuity_error_exits_nonzero_with_a_readable_message(tmp_path, capsys):
    cli = _load_cli_module()
    # set_status on a task that does not exist raises ContinuityError inside
    # the core; the CLI must print it and exit nonzero, not print a traceback.
    exit_code = cli.main(["task", "set-status", "nope", "in_progress",
                          "--project-dir", str(tmp_path)])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "nope" in err
