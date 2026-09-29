# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Direct-use CLI over core.continuity, for a controller session at a context
reset -- mirrors how packaging/check_branding.py and
studio/install_llama_prebuilt.py are both invoked directly with the venv
Python, never imported as a package.

    C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe studio/backend/tools/continuity_cli.py status --project-dir .
    ... status
    ... task add <id> --title "..." [--depends-on id1,id2] [--criteria "c1;c2"]
    ... task set-status <id> <status>
    ... error record "<what was tried>" "<why it failed>" [--symptom "<tag>"]
    ... error check "<symptom query>"
    ... checkpoint "<note>"
    ... validate
    ... repair

Errors from core.continuity (ContinuityError) are printed to stderr and the
process exits 1 -- never a raw traceback, which is unreadable at 2am during an
actual context reset.
"""

from __future__ import annotations

import argparse
import os
import sys

# This script is invoked directly (python continuity_cli.py ...), so
# studio/backend needs to be on sys.path for `core.continuity` to import --
# it is not installed as a package. Inserted once, at the top, before any
# core.* import below.
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# Where `error record` falls back to if errors.jsonl cannot be written. Known
# only here, not in core.continuity: the CLI runs as the user on their own
# machine, so a note in their own .remember is a safety net; the app-side tool
# is model-driven and passes no fallback at all (see record_error). Tests
# monkeypatch this to a tmp_path file and never touch the real one.
_REMEMBER_NOW_PATH = os.path.expanduser(r"~\odysseus\.remember\now.md")


def _default_project_dir() -> str:
    return os.getcwd()


def main(argv: list[str]) -> int:
    # --project-dir is declared on each LEAF subparser (status, validate,
    # repair, init, checkpoint, task-add, task-set-status, error-record,
    # error-check) through this
    # shared parent, rather than once on the top-level parser the way a
    # first draft of this file tried it. argparse's subparsers action
    # consumes everything after the verb token as that subparser's own
    # argv, so a flag declared only on the top-level parser is rejected as
    # "unrecognized arguments" the moment it appears after the verb -- which
    # is how this file's own usage docstring above, every test in
    # test_continuity_cli.py, and Step 5's manual verification all place it
    # (`status --project-dir .`, never `--project-dir . status`). Declaring
    # it on each leaf, and NOT on the intermediate "task"/"error" parsers, also
    # sidesteps the opposite argparse footgun: when a parent parser and a
    # child parser both default the same dest, the child's default silently
    # overwrites whatever value the parent already parsed.
    common = argparse.ArgumentParser(add_help = False)
    common.add_argument("--project-dir", default = None)

    parser = argparse.ArgumentParser(prog = "continuity_cli")
    sub = parser.add_subparsers(dest = "verb")

    sub.add_parser("status", parents = [common])
    sub.add_parser("validate", parents = [common])
    sub.add_parser("repair", parents = [common])

    init = sub.add_parser("init", parents = [common])
    init.add_argument("project")
    init.add_argument("--phase", default = "")

    cp = sub.add_parser("checkpoint", parents = [common])
    cp.add_argument("note")

    task = sub.add_parser("task")
    task_sub = task.add_subparsers(dest = "task_verb")
    add = task_sub.add_parser("add", parents = [common])
    add.add_argument("id")
    add.add_argument("--title", default = None)
    add.add_argument("--depends-on", default = "")
    add.add_argument("--criteria", default = "")
    set_status = task_sub.add_parser("set-status", parents = [common])
    set_status.add_argument("id")
    set_status.add_argument("status")

    error = sub.add_parser("error")
    error_sub = error.add_subparsers(dest = "error_verb")
    record = error_sub.add_parser("record", parents = [common])
    record.add_argument("what_tried")
    record.add_argument("why_failed")
    record.add_argument("--symptom", default = "")
    check = error_sub.add_parser("check", parents = [common])
    check.add_argument("symptom_query")

    # argparse's own parser.error() (an unknown verb, a missing required
    # positional, or any other bad argv) prints a usage message to stderr
    # and calls sys.exit(2) -- it does not raise a catchable exception by
    # default. Left alone, that SystemExit blows straight through this
    # function (main() would never return an int) and through cli.main(...)
    # in the tests that drive this as a function rather than a subprocess,
    # turning "assert exit_code != 0" into an unhandled SystemExit instead
    # of a clean assertion. Caught here and converted to a return value so
    # main() always returns an int, the same contract every other path
    # through this function keeps.
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    # getattr, not args.project_dir: an unrecognized/absent verb returns
    # above via the SystemExit branch, but a syntactically valid parse with
    # no verb at all (dest="verb" is not marked required) reaches here with
    # no subparser having run, and neither the top-level parser nor "task"
    # itself declares --project-dir -- so the attribute would not exist.
    project_dir = getattr(args, "project_dir", None) or _default_project_dir()

    from core.continuity import (
        add_task, checkpoint, load_state, load_tasks, prior_failures, record_error,
        repair, set_task_status, validate, write_state,
    )
    from core.continuity.render import render_context_summary
    from core.continuity.schemas import ContinuityError, ProjectState, Task

    try:
        if args.verb == "init":
            # The one gap the tool deliberately leaves to a human: nothing in
            # continuity_task (the app-side tool) or this CLI's other verbs can
            # ever create the FIRST project_state.json -- update_state requires
            # one to already exist, and write_state itself has no other caller.
            # Refuses rather than overwrites: init is a one-time bootstrap, and
            # an accidental second run must not silently erase real progress
            # (completed, in_progress, blocking_issues, ...).
            if load_state(project_dir) is not None:
                print(f"Error: project_state.json already exists at {project_dir!r} "
                      "-- init refuses to overwrite it.", file = sys.stderr)
                return 1
            write_state(project_dir, ProjectState(
                schema_version = 1, project = args.project, status = "active",
                current_phase = args.phase, current_task = None,
                completion_percent = 0, last_checkpoint = None,
            ))
            print(f"Initialized {args.project!r} at {project_dir!r}.")
            return 0

        if args.verb == "status":
            summary = render_context_summary(project_dir)
            if load_state(project_dir) is None:
                # Mirrors the app-side tool's own "status" action
                # (core/continuity/__init__.py's _execute): a project that
                # has only ever had `task add` called, with no `init`, has
                # no project_state.json, so render_context_summary's early
                # return would otherwise hide every task this CLI has
                # recorded. Fall back to a plain task list so status still
                # reflects what was tracked.
                tasks = load_tasks(project_dir).tasks
                if tasks:
                    lines = [summary, "", "## Tasks"]
                    lines += [f"- [{t.status}] {t.id}: {t.title}" for t in tasks]
                    summary = "\n".join(lines)
            print(summary)
            return 0

        if args.verb == "validate":
            problems = validate(project_dir)
            if not problems:
                print("No problems found.")
                return 0
            for p in problems:
                print(f"- {p}")
            return 1

        if args.verb == "repair":
            actions = repair(project_dir)
            for a in actions:
                print(f"- {a}")
            return 0

        if args.verb == "checkpoint":
            name = checkpoint(project_dir, args.note)
            print(f"Checkpointed as {name}.")
            return 0

        if args.verb == "task":
            if args.task_verb == "add":
                depends = [d for d in args.depends_on.split(",") if d]
                criteria = [c for c in args.criteria.split(";") if c]
                add_task(project_dir, Task(
                    id = args.id, title = args.title or args.id, status = "pending",
                    depends_on = depends, acceptance_criteria = criteria,
                ))
                print(f"Added task {args.id!r}.")
                return 0
            if args.task_verb == "set-status":
                set_task_status(project_dir, args.id, args.status)
                print(f"Task {args.id!r} is now {args.status!r}.")
                return 0
            print(f"Unknown task verb: {args.task_verb!r}", file = sys.stderr)
            return 2

        if args.verb == "error":
            if args.error_verb == "record":
                recorded = record_error(
                    project_dir, what_tried = args.what_tried, why_failed = args.why_failed,
                    symptom = args.symptom, fallback_path = _REMEMBER_NOW_PATH,
                )
                if not recorded:
                    print("Error: could not record this -- the write to .ai/logs/errors.jsonl "
                          f"failed, and so did the fallback note at {_REMEMBER_NOW_PATH!r}.",
                          file = sys.stderr)
                    return 1
                print("Recorded.")
                return 0
            if args.error_verb == "check":
                hits = prior_failures(project_dir, args.symptom_query)
                if not hits:
                    print("No prior recorded failures match that.")
                    return 0
                for h in hits:
                    print(f"- {h.what_tried} -- {h.why_failed}")
                return 0
            print(f"Unknown error verb: {args.error_verb!r}", file = sys.stderr)
            return 2

        print(f"Unknown verb: {args.verb!r}", file = sys.stderr)
        return 2
    except ContinuityError as exc:
        print(f"Error: {exc}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
