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


WORKSPACE_ROOT_KEY = "workspace_root"


def _read_setting(key: str, fallback = None):
    """Indirection so tests can substitute storage without a database."""
    from storage.studio_db import get_app_setting
    return get_app_setting(key, fallback)


def _write_setting(mapping: dict) -> None:
    from storage.studio_db import upsert_app_settings
    upsert_app_settings(mapping)


# The last value a read actually produced. "Could not read the setting" is not
# the same answer as "the user has not chosen a folder", and callers cannot tell
# them apart from a None: a single SQLITE_BUSY would otherwise move a live chat
# out of the user's folder and into an empty sandbox mid-conversation.
_last_known_root: "str | None" = None


def get_global_root() -> "str | None":
    """The global default workspace root, or None when unset or unusable.

    Returns None rather than a path in three cases that would each be worse
    than falling through to the sandbox:
      * unset -- the feature is opt-in, so nothing set means today's behaviour
      * blank -- an empty string would resolve to the process cwd
      * missing on disk -- _get_workdir calls makedirs() on whatever it returns,
        so a stale value would silently recreate a folder the user deleted

    A failure to READ the setting is none of those, and is answered with the
    last value this process did read (None until it has read one). Only a
    successful read updates that, so a folder the user genuinely cleared or
    deleted still falls through on the very next call.
    """
    global _last_known_root
    try:
        raw = _read_setting(WORKSPACE_ROOT_KEY, None)
    except Exception:  # noqa: BLE001 - a storage hiccup is not "unset"
        # The cached path itself can have been deleted since it was last
        # confirmed. Handing it back unchecked is the one outcome this
        # function promises never to produce -- _get_workdir calls
        # makedirs() on whatever it gets back.
        if _last_known_root is not None and not os.path.isdir(_last_known_root):
            return None
        return _last_known_root
    if not isinstance(raw, str) or not raw.strip():
        _last_known_root = None
        return None
    try:
        resolved = _real(raw.strip())
        usable = os.path.isdir(resolved)
    except (OSError, ValueError):
        # A stored value that cannot even be resolved (an embedded NUL, say) is
        # "unusable", which is one of the documented None cases -- not a read
        # failure, so it clears the cached value rather than keeping it.
        _last_known_root = None
        return None
    if not usable:
        _last_known_root = None
        return None
    _last_known_root = resolved
    return resolved


def _invalidate_workdir_cache() -> None:
    """Drop every cached sandbox resolution in core/inference/tools.py.

    _get_workdir and resolve_sandbox_workdir consult that cache in opposite
    orders relative to this module's root: the former checks it before the
    global root, the latter after. Nothing else clears the cache when the
    global root changes, so a session that ran a tool before the folder was
    chosen would otherwise keep writing to its old sandbox while every
    read-only route (download, listing, "open folder") already serves the
    new one -- for the rest of the process's life.

    Imported lazily to match _global_workspace_root's own lazy import of this
    module: that import exists to avoid a circular import, and a module-level
    import here would create the same cycle in the other direction. Wrapped
    because the setting write above has already succeeded by the time this
    runs -- losing the cache is only a missed optimisation (_get_workdir
    re-resolves and re-creates idempotently, project entries included), and
    must never turn a saved setting into a failed one.

    Clears the whole cache rather than filtering to sandbox-contained
    entries: a project workdir re-resolves to the identical path (it does not
    depend on the global root), so dropping it costs one redundant lookup and
    leaves nothing stale, while a narrower filter would have to reimplement
    tools.py's own containment/ownership rules here and risk drifting from
    them.
    """
    try:
        from core.inference import tools
        tools._workdirs.clear()
    except Exception:  # noqa: BLE001 - the write already succeeded
        pass


def set_global_root(path: "str | None") -> "str | None":
    """Store the global default. ``None`` or blank clears it. Never refuses a
    path -- classification is advisory and belongs to the caller."""
    if path is None or not str(path).strip():
        _write_setting({WORKSPACE_ROOT_KEY: None})
        _invalidate_workdir_cache()
        return None
    resolved = _real(str(path).strip())
    _write_setting({WORKSPACE_ROOT_KEY: resolved})
    _invalidate_workdir_cache()
    return resolved
