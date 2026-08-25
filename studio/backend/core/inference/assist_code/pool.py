# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A bounded pool of language-server sessions keyed by (language, root).

Persistent because starting a server is expensive -- typescript-language-server
takes seconds to index and rust-analyzer/clangd take tens of seconds, so a
server-per-call design would make every tool call unusable.

Per-key locking follows ``mcp_client._StdioKeyLock`` (:540): without it, two
threads asking for the same workspace at once each spawn a server and one is
orphaned. ``execute_tool`` runs on a daemon worker thread
(``tool_stream_exec.py:161-176``), so concurrent calls are real.
"""
import atexit
import logging
import os
import threading
import time

try:
    from loggers import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - outside the app
    logger = logging.getLogger(__name__)

MAX_SESSIONS = 4
IDLE_TIMEOUT_SECONDS = 600

_lock = threading.Lock()
_key_locks = {}
_sessions = {}      # key -> session
_last_used = {}     # key -> monotonic timestamp


def _key(language, root):
    return (language, os.path.abspath(root))


def _key_lock(key):
    with _lock:
        lk = _key_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _key_locks[key] = lk
        return lk


def _close_quietly(session):
    try:
        session.close()
    except Exception:
        pass


def _evict_if_needed():
    """Caller holds no lock. Drops LRU sessions until under the cap."""
    while True:
        with _lock:
            if len(_sessions) <= MAX_SESSIONS:
                return
            victim_key = min(_last_used, key = lambda k: _last_used.get(k, 0))
            victim = _sessions.pop(victim_key, None)
            _last_used.pop(victim_key, None)
        if victim is not None:
            logger.info("assist_code: evicting idle language server for %s", victim_key)
            _close_quietly(victim)


def acquire(language, root, factory, *, start_timeout = 60.0):
    """Return a live session for (language, root), creating one if needed.

    A dead session is replaced rather than handed back. A server that fails to
    start is NOT cached -- caching a corpse turns one bad start into a
    permanent failure for that workspace.
    """
    key = _key(language, root)
    with _key_lock(key):
        with _lock:
            existing = _sessions.get(key)
        if existing is not None:
            if existing.is_healthy():
                with _lock:
                    _last_used[key] = time.monotonic()
                return existing
            with _lock:
                _sessions.pop(key, None)
                _last_used.pop(key, None)
            _close_quietly(existing)

        session = factory(os.path.abspath(root))
        session.start(timeout = start_timeout)   # SessionStartFailed propagates
        with _lock:
            _sessions[key] = session
            _last_used[key] = time.monotonic()
    _evict_if_needed()
    return session


def reap_idle():
    now = time.monotonic()
    stale = []
    with _lock:
        for key, used in list(_last_used.items()):
            if now - used >= IDLE_TIMEOUT_SECONDS:
                stale.append((key, _sessions.pop(key, None)))
                _last_used.pop(key, None)
    for key, session in stale:
        if session is not None:
            logger.info("assist_code: reaping idle language server for %s", key)
            _close_quietly(session)


def stats():
    with _lock:
        return {"live": len(_sessions), "keys": sorted(str(k) for k in _sessions)}


def shutdown_all():
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
        _last_used.clear()
    for s in sessions:
        _close_quietly(s)


def _atexit_cleanup():
    """Terminate every server at interpreter exit.

    Nothing here may report through logging. By the time atexit runs, the
    streams handlers write to can already be closed, and a write then fails --
    which is how the equivalent bug was found in llama_cpp.py:20925, as
    unrelated tracebacks printed after the test summary. Two mechanisms are
    needed together: raiseExceptions covers the stdlib loggers other libraries
    install, and the bare except covers this module's structlog PrintLogger,
    which does not consult raiseExceptions at all.
    """
    raise_exceptions = logging.raiseExceptions
    logging.raiseExceptions = False
    try:
        shutdown_all()
    except Exception:
        pass
    finally:
        logging.raiseExceptions = raise_exceptions


atexit.register(_atexit_cleanup)
