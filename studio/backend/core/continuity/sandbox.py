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


def confine_project_dir(session_id: str | None) -> str:
    """The sandbox workdir continuity_task operates on for this session.
    _get_workdir is None-safe (session_id or an anonymous key), so there is no
    session_id-omitted escape from confinement."""
    from core.inference import tools as _tools

    return _tools._get_workdir(session_id)
