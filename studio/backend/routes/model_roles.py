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


# Paths are relative: main.py mounts this router at prefix "/api/model-roles",
# the way it mounts routes/tool_audit.py at "/api/tool-audit".
@router.get("")
def read_model_roles(current_subject: str = Depends(get_current_subject)):
    from core.inference import model_roles

    out = {}
    for name, binding in model_roles.bindings().items():
        state = model_roles.availability(name)
        out[name] = {
            "model": binding.model,
            "overrides": binding.overrides,
            "state": state.state,
            "detail": state.detail,
        }
    return {"roles": out, "defaults": list(model_roles.DEFAULT_ROLE_NAMES)}


@router.put("")
def write_model_roles(
    payload: RolesPayload,
    current_subject: str = Depends(get_current_subject),
):
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
