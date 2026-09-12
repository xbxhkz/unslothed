# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only API over the tool audit log.

Read-only on purpose. Records are evidence; an endpoint that edited or deleted
them would undermine the point of keeping them. Pruning is a retention policy
inside the storage layer, and it records itself.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth.authentication import get_current_subject
from core.inference import tool_audit
from storage import tool_audit_db

router = APIRouter()


class AuditStatus(BaseModel):
    degraded: bool
    failed_writes: int


@router.get("/entries")
def list_entries(
    limit: int = Query(100, ge = 1, le = 1000),
    offset: int = Query(0, ge = 0),
    tool_name: Optional[str] = None,
    session_id: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    tool_audit_db.maybe_prune()
    entries = tool_audit_db.query_entries(
        limit = limit,
        offset = offset,
        tool_name = tool_name,
        session_id = session_id,
    )
    return {"entries": entries, "count": len(entries)}


@router.get("/entries/{entry_id}")
def get_entry(
    entry_id: int,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    entry = tool_audit_db.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code = 404, detail = "No such audit entry")
    return entry


@router.get("/status", response_model = AuditStatus)
def status(
    current_subject: str = Depends(get_current_subject),
) -> AuditStatus:
    failed = tool_audit.degraded_count()
    return AuditStatus(degraded = failed > 0, failed_writes = failed)
