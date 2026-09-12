# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Record every tool invocation.

Installed over execute_tool by a definition-time shadow at the end of tools.py,
so all three callers -- studio_tool_loop, safetensors_agentic, and llama_cpp
(which this fork does not edit) -- are covered by one hook.

THE INVARIANT: auditing must never break a tool call. Every recording path is
guarded, and the guard is a BARE except: this project has established that
"never raises" needs one, because OverflowError and unhashable types slip past
`except Exception` in practice.

Silence is its own failure, so failures increment a counter that the API
surfaces as "audit degraded". An audit log that quietly stops working leaves
holes with no indication.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Per side. A 50 KB result stores its first and last 4 KB; result_bytes keeps the
# true length so truncation is always detectable.
RESULT_CAP_BYTES = 4096

_degraded = 0


def degraded_count() -> int:
    return _degraded


def reset_degraded_for_tests() -> None:
    global _degraded
    _degraded = 0


def _note_failure(what: str, exc: BaseException) -> None:
    global _degraded
    _degraded += 1
    logger.warning("tool audit %s failed (%s); tool execution unaffected", what, exc)


def _split_result(text: str) -> tuple[str, str, int, str]:
    """(head, tail, true_byte_length, sha256_of_redacted_text)."""
    from core.inference.tool_audit import redaction

    safe, _ = redaction.redact_text(text)
    raw = text.encode("utf-8", errors = "replace")
    # Hash the REDACTED payload, not the original: hashing the original would let
    # a short secret be confirmed by brute force against the stored digest.
    digest = hashlib.sha256(safe.encode("utf-8", errors = "replace")).hexdigest()
    if len(safe) <= RESULT_CAP_BYTES * 2:
        return safe, "", len(raw), digest
    return safe[:RESULT_CAP_BYTES], safe[-RESULT_CAP_BYTES:], len(raw), digest


def around(fn: Callable[..., str], *args: Any, **kwargs: Any) -> str:
    """Call ``fn`` and record the invocation. Returns ``fn``'s result unchanged."""
    row_id = None
    started = time.monotonic()
    try:
        from core.inference.tool_audit import redaction
        from storage import tool_audit_db

        # All three callers pass name and arguments positionally, then **kwargs.
        name = args[0] if args else kwargs.get("name")
        arguments = args[1] if len(args) > 1 else kwargs.get("arguments")
        args_json, redacted = redaction.redact_arguments(arguments)
        row_id = tool_audit_db.record_start(
            tool_name = str(name),
            arguments_json = args_json,
            paths_json = redaction.extract_paths(arguments),
            redacted = redacted,
            session_id = kwargs.get("session_id"),
            thread_id = kwargs.get("thread_id"),
            disable_sandbox = bool(kwargs.get("disable_sandbox", False)),
        )
    except BaseException as exc:  # noqa: BLE001 - never-raises; see module docstring
        _note_failure("start", exc)

    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:
        _finish(row_id, started, outcome = "error", text = "", error = repr(exc))
        raise
    _finish(row_id, started, outcome = "ok", text = result if isinstance(result, str) else str(result))
    return result


def _finish(row_id, started: float, *, outcome: str, text: str, error: str | None = None) -> None:
    if row_id is None:
        return
    try:
        from storage import tool_audit_db

        head, tail, size, digest = _split_result(text)
        tool_audit_db.record_finish(
            row_id,
            outcome = outcome,
            duration_ms = int((time.monotonic() - started) * 1000),
            result_head = head,
            result_tail = tail,
            result_bytes = size,
            result_sha256 = digest,
            error_text = error,
        )
    except BaseException as exc:  # noqa: BLE001 - never-raises; see module docstring
        _note_failure("finish", exc)
