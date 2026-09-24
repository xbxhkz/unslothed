# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read and replace role bindings.

Follows routes/tool_audit.py: a fork-owned router, every endpoint gated by
get_current_subject. Bindings name models and launch settings, so this is not
public.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.authentication import get_current_subject

router = APIRouter()


class RolesPayload(BaseModel):
    roles: Any


# Stored, echoed back, and applied by nothing. Delegation loads a model through
# the route's own auto-switch, which applies that model's SAVED launch settings via
# model_override_load_kwargs; a second conversion path here would silently disagree
# with it. So `overrides` is a preference this API keeps, not a setting that takes
# effect -- and a user who PUTs {"n_ctx": 16384} beside `state: "ready"` has no way
# to tell that from the response unless it says so. Hence the flag on every role and
# the sentence in both descriptions: an ignored field that looks accepted is worse
# than an absent one.
OVERRIDES_NOT_APPLIED = (
    "Role overrides are stored and returned but are NOT applied: a model is loaded "
    "with its own saved launch settings, exactly as it would be by an API request "
    "naming it. Change the model's launch settings to change how it loads."
)


# Paths are relative: main.py mounts this router at prefix "/api/model-roles",
# the way it mounts routes/tool_audit.py at "/api/tool-audit".
@router.get("", description = "Read every role binding and its availability. " + OVERRIDES_NOT_APPLIED)
def read_model_roles(current_subject: str = Depends(get_current_subject)):
    """Every bound role, its model, its availability, and its stored overrides.

    `overrides_applied` is False on every role and is not a per-role setting: see
    OVERRIDES_NOT_APPLIED above for why nothing reads them.
    """
    from core.inference import model_roles

    out = {}
    for name, binding in model_roles.bindings().items():
        state = model_roles.availability(name)
        out[name] = {
            "model": binding.model,
            "overrides": binding.overrides,
            # Carried per role rather than once at the top level: a client reading
            # one role's entry sees it without having to know to look elsewhere.
            "overrides_applied": False,
            "state": state.state,
            "detail": state.detail,
        }
    return {
        "roles": out,
        "defaults": list(model_roles.DEFAULT_ROLE_NAMES),
        "overrides_note": OVERRIDES_NOT_APPLIED,
    }


@router.put("", description = "Replace every role binding. " + OVERRIDES_NOT_APPLIED)
def write_model_roles(
    payload: RolesPayload,
    current_subject: str = Depends(get_current_subject),
):
    """Replace the whole set of role bindings and return the new state.

    `overrides` is validated and stored, and then nothing reads it -- see
    OVERRIDES_NOT_APPLIED above. The response says so on every role.
    """
    from core.inference import model_roles
    from core.inference.model_roles import storage

    roles = payload.roles
    if not isinstance(roles, dict):
        raise HTTPException(status_code = 400, detail = "roles must be an object")
    cleaned: dict[str, dict] = {}
    for name, raw in roles.items():
        role = model_roles.normalize_role(name)
        if not role:
            raise HTTPException(status_code = 400, detail = "a role name cannot be empty")
        if not isinstance(raw, dict):
            raise HTTPException(status_code = 400, detail = f"{role}: binding must be an object")
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HTTPException(status_code = 400, detail = f"{role}: a model is required")
        overrides = raw.get("overrides")
        if overrides is not None and not isinstance(overrides, dict):
            raise HTTPException(status_code = 400, detail = f"{role}: overrides must be an object")
        cleaned[role] = {"model": model.strip(), "overrides": overrides or {}}
    storage.set_role_bindings(cleaned)
    return read_model_roles(current_subject = current_subject)
