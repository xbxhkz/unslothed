# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where continuity_task is allowed to read and write, for a given session.

For this tool the "project" IS the conversation's own sandbox workdir -- there
is no separate project_dir argument for a model to name, which is what makes a
path-escape attempt structurally impossible rather than merely checked for.
Reuses core.inference.tools._get_workdir exactly as
core/inference/assist_vision/paths.py:resolve_image_bytes does; this module
does not invent a second confinement policy.

Imported lazily inside the function, matching paths.py's own stated reason:
tools.py is large and will eventually import continuity back (to register
continuity_task), so a module-level import here would be circular.
"""

from __future__ import annotations

import os
import re


def confine_project_dir(session_id: str | None, thread_id: str | None = None) -> str:
    """The sandbox workdir continuity_task operates on for this conversation.

    Namespaced by thread_id when one is available: _get_workdir's root can be
    a SHARED, project-wide directory (a project session, or the global-
    workspace setting), not a per-chat one -- without this, two chats sharing
    that root would collide on the same task ids and the same
    project_state.json. Falls back to the bare session workdir when
    thread_id is unavailable, which is still confined and safe, just not
    further namespaced (this is the CLI's case -- it has no thread_id
    concept and always names --project-dir explicitly instead).

    thread_id is sanitized to a safe directory-component shape before use,
    the same defensive posture the rest of this package takes with any
    caller-influenced string that becomes a path segment.
    """
    from core.inference import tools as _tools

    workdir = _tools._get_workdir(session_id)
    if not thread_id:
        return workdir
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(thread_id))[:128]
    return os.path.join(workdir, "continuity-threads", safe)
