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
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Per side. A 50 KB result stores its first and last 4 KB; result_bytes keeps the
# true length so truncation is always detectable.
RESULT_CAP_BYTES = 4096

# Redaction layer 1 (see the design spec's Redaction section): what a withheld
# row's result_head reads as. Never the real content -- result_bytes still
# carries the true length, so a withheld row is never mistaken for an empty one.
WITHHELD_MARKER = "[result withheld: credential-referencing MCP call]"

_degraded = 0
# Tool calls run concurrently -- multiple agentic loops share this process --
# so the counter's increment must not race and silently lose updates. An
# inaccurate degraded count is a partial instance of the exact failure mode
# the counter exists to prevent.
_degraded_lock = threading.Lock()


def degraded_count() -> int:
    return _degraded


def reset_degraded_for_tests() -> None:
    global _degraded
    with _degraded_lock:
        _degraded = 0


def _note_failure(what: str, exc: BaseException) -> None:
    global _degraded
    with _degraded_lock:
        _degraded += 1
    logger.warning("tool audit %s failed (%s); tool execution unaffected", what, exc)


def _mcp_result_must_be_withheld(name: Any, arguments: Any) -> bool:
    """Redaction layer 1: true when ``name`` is an MCP-prefixed tool call whose
    arguments name a credential path, a secret environment variable, or a
    cloud-metadata host.

    Reuses upstream's own classifier (``tools.py``'s
    ``_mcp_arguments_reference_sensitive``) rather than writing a second one,
    so the two checks cannot drift apart. Only MCP tools are in scope -- a
    non-MCP tool call is always unaffected, since upstream's classifier was
    never written to judge any other tool's argument shapes.

    Fails CLOSED: any exception raised while consulting the classifier is
    treated as "withhold", never as "safe to store".
    """
    try:
        from core.inference.tools import MCP_TOOL_PREFIX

        if not isinstance(name, str) or not name.startswith(MCP_TOOL_PREFIX):
            return False
    except BaseException:  # noqa: BLE001 - can't classify; nothing to withhold from
        return False

    try:
        from core.inference.tools import _mcp_arguments_reference_sensitive

        return bool(_mcp_arguments_reference_sensitive(arguments))
    except BaseException:  # noqa: BLE001 - fail closed: any doubt means withhold
        return True


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
    withhold = False
    started = time.monotonic()
    try:
        from core.inference.tool_audit import redaction
        from storage import tool_audit_db

        # All three callers pass name and arguments positionally, then **kwargs.
        name = args[0] if args else kwargs.get("name")
        arguments = args[1] if len(args) > 1 else kwargs.get("arguments")
        withhold = _mcp_result_must_be_withheld(name, arguments)
        args_json, redacted = redaction.redact_arguments(arguments)
        row_id = tool_audit_db.record_start(
            tool_name = str(name),
            arguments_json = args_json,
            paths_json = redaction.extract_paths(arguments),
            # Withholding must be visible on the row even when the arguments
            # themselves redacted clean -- a withheld result is never "clean".
            redacted = redacted or withhold,
            session_id = kwargs.get("session_id"),
            thread_id = kwargs.get("thread_id"),
            disable_sandbox = bool(kwargs.get("disable_sandbox", False)),
        )
    except BaseException as exc:  # noqa: BLE001 - never-raises; see module docstring
        _note_failure("start", exc)

    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:
        # exc itself is passed through, not repr(exc): the coercion must happen
        # inside _finish's own guard, or an exception whose __repr__ raises
        # would replace the caller's real exception with the audit's.
        _finish(row_id, started, outcome = "error", result = "", error = exc, withhold = withhold)
        raise
    _finish(row_id, started, outcome = "ok", result = result, withhold = withhold)
    return result


def _finish(
    row_id,
    started: float,
    *,
    outcome: str,
    result: Any,
    error: BaseException | None = None,
    withhold: bool = False,
) -> None:
    if row_id is None:
        return
    try:
        from core.inference.tool_audit import redaction
        from storage import tool_audit_db

        error_text = None
        if error is not None:
            # repr() happens INSIDE this guard, mirroring the result coercion
            # below: an exception whose __repr__ raises must not replace the
            # caller's real exception with a failure from formatting it for
            # storage. Errors are kept UNTRUNCATED (spec: "errors kept whole"
            # means whole, not unredacted) -- arguments get scrubbed but
            # tracebacks routinely embed the very values that were just
            # redacted, so the same scrub runs here before storage.
            error_text, _ = redaction.redact_text(repr(error))

        # The str() coercion happens INSIDE this guard, not in around()'s frame:
        # a successful tool call must not be broken by a result whose __str__
        # raises (an OverflowError-shaped object is exactly the precedent this
        # module's docstring cites).
        text = result if isinstance(result, str) else str(result)
        if withhold:
            # Redaction layer 1: never store what an MCP server returned for a
            # credential-referencing call. result_bytes still records the TRUE
            # size of what was suppressed, so truncation stays detectable even
            # though the content itself never reaches the row.
            raw = text.encode("utf-8", errors = "replace")
            head, tail = WITHHELD_MARKER, ""
            size = len(raw)
            digest = hashlib.sha256(WITHHELD_MARKER.encode("utf-8")).hexdigest()
        else:
            head, tail, size, digest = _split_result(text)
        tool_audit_db.record_finish(
            row_id,
            outcome = outcome,
            duration_ms = int((time.monotonic() - started) * 1000),
            result_head = head,
            result_tail = tail,
            result_bytes = size,
            result_sha256 = digest,
            error_text = error_text,
        )
    except BaseException as exc:  # noqa: BLE001 - never-raises; see module docstring
        _note_failure("finish", exc)
