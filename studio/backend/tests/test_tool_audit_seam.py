# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audit hook must live INSIDE tools.py, wrapping execute_tool.

execute_tool has three callers and one of them is core/inference/llama_cpp.py,
which this fork does not edit. A hook at the call sites would therefore miss a
whole agentic path. These tests exist to stop a later 'simplification' moving it
there, and to stop the functools.wraps disappearing.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from core.inference import tools

_TOOLS_PY = Path(inspect.getfile(tools))


def test_execute_tool_is_wrapped_by_the_audit_shadow():
    assert hasattr(tools, "_execute_tool_unaudited"), (
        "the definition-time shadow is gone; execute_tool is no longer audited"
    )
    assert tools.execute_tool is not tools._execute_tool_unaudited


def test_the_wrapper_preserves_the_original_signature():
    """functools.wraps keeps the original parameters visible through inspect.signature.

    This is the live control for the decorator: remove @functools.wraps from the
    shadow in tools.py and this test fails, because inspect.signature(execute_tool)
    collapses to just (*args, **kwargs) and every named parameter -- including
    "name" -- vanishes. Verified by hand: deleting the decorator makes this test
    fail with exactly that assertion.

    Note this is NOT about kwarg forwarding -- see
    test_accepts_kwarg_still_sees_the_forwarded_kwargs's docstring for why
    forwarding is unaffected either way.
    """
    params = inspect.signature(tools.execute_tool).parameters
    for expected in ("name", "arguments", "session_id", "thread_id",
                     "disable_sandbox", "conversation_branch", "conversation_budget_tokens"):
        assert expected in params, f"{expected} vanished from the public signature"


def test_accepts_kwarg_still_sees_the_forwarded_kwargs():
    """Corroborating only -- NOT a control for functools.wraps.

    This asserts that forwarding works, but it passes with or without
    @functools.wraps: accepts_kwarg (core/inference/tool_stream_exec.py:35)
    returns True for any callable that takes **kwargs, and the shadow in
    tools.py always does, decorator or not. Verified by hand: removing
    @functools.wraps does not make this test fail.
    functools.wraps is real -- for introspection, tracebacks, and API docs, see
    the comment above the shadow in tools.py and
    test_the_wrapper_preserves_the_original_signature -- it is just not what
    this particular test is evidence of.
    """
    from core.inference.tool_stream_exec import accepts_kwarg

    assert accepts_kwarg(tools.execute_tool, "conversation_branch") is True
    assert accepts_kwarg(tools.execute_tool, "conversation_budget_tokens") is True


def test_the_hook_is_in_tools_py_and_not_at_the_call_sites():
    """AST check: the shadow must be a module-level assignment in tools.py."""
    tree = ast.parse(_TOOLS_PY.read_text(encoding = "utf-8"))
    names = {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "_execute_tool_unaudited" in names, (
        "the audit shadow is not a module-level assignment in tools.py -- if the "
        "hook moved to the call sites, the llama_cpp.py path is no longer audited"
    )


def test_the_seam_stays_additive():
    """tools.py must never lose a line to this fork."""
    import subprocess

    repo = _TOOLS_PY.parents[4]
    base = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.strip()
    stat = subprocess.run(
        ["git", "diff", "--numstat", base, "--", "studio/backend/core/inference/tools.py"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.split()
    assert stat, "no diff recorded for tools.py"
    insertions, deletions = int(stat[0]), int(stat[1])
    assert deletions == 0, f"tools.py lost {deletions} line(s) to the fork; the seam must be additive"
    assert insertions <= 80, f"seam grew to {insertions} insertions; budget is ~73"
