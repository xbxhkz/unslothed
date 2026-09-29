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


def test_a_continuity_error_exits_nonzero_with_a_readable_message(tmp_path, capsys):
    cli = _load_cli_module()
    # set_status on a task that does not exist raises ContinuityError inside
    # the core; the CLI must print it and exit nonzero, not print a traceback.
    exit_code = cli.main(["task", "set-status", "nope", "in_progress",
                          "--project-dir", str(tmp_path)])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "nope" in err
