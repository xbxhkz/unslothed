# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The directory the AI's tools work in, when the user has chosen one.

Studio resolves a chat's working directory through core/inference/tools.py's
_get_workdir. This module supplies the two inputs that can override the
per-session sandbox -- a global default, and (via storage/studio_db.py) a
per-project root -- plus the advisory classification the UI shows before a
choice is committed.

The policy is WARN ONLY. Nothing here refuses a path, and there is no rejection
channel in the return type to refuse through. That is deliberate: the owner
chose full read/write access to any directory, and a classifier that quietly
declined would be a different feature wearing this one's name.

The consequence is worth stating where someone changing this file will read it:
whatever root is chosen becomes what upstream's `terminal` tool runs inside.
The ten tools this fork adds confine themselves to the workdir; `terminal` does
not confine, it inhabits. The warnings below exist because that is the whole
exposure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

WARN_STUDIO_DATA = "studio_data"
WARN_INSTALL_DIR = "install_dir"
WARN_SYSTEM_DIR = "system_dir"
WARN_BROAD = "broad"


@dataclass(frozen = True)
class RootWarning:
    code: str
    message: str


def _real(path: str) -> str:
    """Fully resolved, so a symlink or `..` cannot dodge a rule by spelling."""
    return os.path.realpath(os.path.expanduser(str(path)))


def _is_within(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or sits inside it.

    Compared as path components, never as a string prefix: `studio-notes`
    starts with `studio` and is a different directory.
    """
    p, r = _real(path), _real(root)
    if p == r:
        return True
    return p.startswith(r.rstrip(os.sep) + os.sep)


def _studio_root() -> str:
    """Studio's own data directory. Patched in tests."""
    from utils.paths import studio_root
    return str(studio_root())


def _install_root() -> str:
    """Where the app itself lives. Frozen, that is the directory holding the
    exe; from source it is the repo checkout. Patched in tests."""
    import sys
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _home() -> str:
    return os.path.expanduser("~")


def _system_roots() -> list[str]:
    roots = [os.path.abspath(os.sep)]
    for var in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)"):
        value = (os.environ.get(var) or "").strip()
        if value:
            roots.append(value)
    return roots


def classify_root(path: str) -> list[RootWarning]:
    """Advisory warnings for a candidate workspace root. Never refuses."""
    out: list[RootWarning] = []
    if not path:
        return out

    try:
        if _is_within(path, _studio_root()):
            out.append(RootWarning(
                WARN_STUDIO_DATA,
                "This is Studio's own data directory. The AI will be able to modify its "
                "auth database, saved models and chat history.",
            ))
    except Exception:
        pass

    try:
        if _is_within(path, _install_root()):
            out.append(RootWarning(
                WARN_INSTALL_DIR,
                "This is where Unslothed is installed. The AI will be able to modify the "
                "application itself.",
            ))
    except Exception:
        pass

    for root in _system_roots():
        try:
            if _real(path) == _real(root):
                out.append(RootWarning(
                    WARN_SYSTEM_DIR,
                    "This is a system directory. Tools have full read and write access here.",
                ))
                break
        except Exception:
            continue

    # Only the directory ITSELF is "broad" -- a folder inside home is an
    # ordinary choice and warning about it would train the user to click past
    # every warning, including the ones above that matter.
    broad = [_home()]
    for name in ("Documents", "Desktop", "Downloads"):
        broad.append(os.path.join(_home(), name))
    for root in broad:
        try:
            if _real(path) == _real(root):
                out.append(RootWarning(
                    WARN_BROAD,
                    "This is a broad location — the AI will see everything inside it.",
                ))
                break
        except Exception:
            continue

    return out
