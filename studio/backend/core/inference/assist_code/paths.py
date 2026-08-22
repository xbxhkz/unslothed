# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Confine a file path or a workspace root to the session's sandbox.

Reuses Studio's own sandbox machinery (``_get_workdir`` / ``_is_outside_workdir``
-- the pair ``edit_file`` resolves through) rather than inventing a second
model. Confinement ALWAYS applies: ``_get_workdir`` is None-safe by Studio's
design (``key = session_id or _ANON_KEY``), so an omitted ``session_id`` still
resolves to a real anonymous sandbox and is still confined to it.

A language server indexes an entire directory tree, so an unconfined workspace
root is a materially worse leak than an unconfined single-file read.

The ``tools`` import is deferred into function bodies: ``tools`` imports this
package back to register the code tools, so module-scope import is circular.
"""
import os

_PROJECT_MARKERS = (
    "package.json", "tsconfig.json", "jsconfig.json",   # JS / TS
    "*.sln", "*.csproj",                                # C#
    ".git",                                             # last resort
)


def _confine(path_value, session_id, label):
    """Shared resolution. Returns (abs_path, workdir, None) or (None, None, err)."""
    if not path_value or not str(path_value).strip():
        return None, None, f"{label} is required"
    raw = os.path.expanduser(str(path_value).strip())
    try:
        from core.inference import tools as _tools
        workdir = _tools._get_workdir(session_id)
        candidate = raw if os.path.isabs(raw) else os.path.join(workdir, raw)
        candidate = os.path.abspath(candidate)
        outside = _tools._is_outside_workdir(candidate, workdir)
    except Exception as e:
        return None, None, f"could not confine {label}: {e}"
    if outside:
        return None, None, (
            f"{label} is outside this conversation's working directory. "
            "Use a path inside it, or a bare filename, which resolves there."
        )
    return candidate, workdir, None


def resolve_file(path_value, *, session_id = None):
    """Resolve to an existing file inside the sandbox. Never raises."""
    candidate, _workdir, err = _confine(path_value, session_id, "path")
    if err:
        return None, err
    if not os.path.exists(candidate):
        return None, f"path not found: {path_value}"
    if not os.path.isfile(candidate):
        return None, f"path is not a file: {path_value}"
    return candidate, None


def resolve_workspace(path_value, *, session_id = None):
    """Resolve to an existing directory inside the sandbox. Never raises."""
    candidate, _workdir, err = _confine(path_value, session_id, "workspace")
    if err:
        return None, err
    if not os.path.exists(candidate):
        return None, f"workspace not found: {path_value}"
    if not os.path.isdir(candidate):
        return None, f"workspace is not a directory: {path_value}"
    return candidate, None


def _has_marker(directory):
    import glob
    for marker in _PROJECT_MARKERS:
        if "*" in marker:
            if glob.glob(os.path.join(directory, marker)):
                return True
        elif os.path.exists(os.path.join(directory, marker)):
            return True
    return False


def workspace_for(file_path, *, session_id = None):
    """Nearest enclosing project root, never climbing above the sandbox.

    Climbing past the workdir would hand the server a root outside the sandbox,
    which is exactly what confinement exists to prevent -- so the workdir is
    both the fallback and the ceiling.
    """
    try:
        from core.inference import tools as _tools
        workdir = os.path.abspath(_tools._get_workdir(session_id))
    except Exception:
        return os.path.dirname(os.path.abspath(file_path))

    current = os.path.dirname(os.path.abspath(file_path))
    while True:
        if not current.startswith(workdir):
            return workdir
        if _has_marker(current):
            return current
        if current == workdir:
            return workdir
        parent = os.path.dirname(current)
        if parent == current:
            return workdir
        current = parent
