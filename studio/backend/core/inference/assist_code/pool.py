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

``lease()`` is the protected way to hold a session across a request. Fix
round 1 found that ``acquire()`` alone has no notion of "a caller is
mid-request against this session": eviction and idle-reaping picked victims
purely by ``last_used``, so a session with a request genuinely in flight
could be, and under review was, evicted and closed out from under its
caller -- who then saw an unrelated ``LspClosed`` instead of its own
request completing. ``mcp_client._evict_stdio_lru_locked`` (:748) is the
model this was missing: it filters eviction candidates to ``in_flight == 0``.
"""
import atexit
import contextlib
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
_in_flight = {}      # key -> active-lease count; guarded by _lock, see lease()


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


def _evict_locked():
    """Caller holds ``_lock``. Trims LRU *idle* sessions to make room for a
    new one the caller is about to insert -- BEFORE that insertion, not
    after.

    This is the detail in ``mcp_client._get_stdio_session`` (around :679)
    that ``_evict_stdio_lru_locked`` (:748) alone doesn't show: eviction
    runs, and only then is the new session published to ``_stdio_sessions``.
    Evicting AFTER insertion (this module's first cut at this) can end up
    choosing the just-inserted session as its own victim, if every
    previously-existing session happens to be in flight -- which defeats
    the entire point of inserting it and was caught by this round's own
    "second acquire while another is in flight, at cap 1" test.

    Skips any key with an active lease (``_in_flight[key] > 0``): a session
    mid-request must never be closed out from under its caller. If every
    session over the cap is in flight, the pool transiently overshoots
    MAX_SESSIONS instead of evicting one of them -- a temporarily oversized
    pool is strictly better than a corrupted in-flight request.

    Returns ``(victims, overshoot)``. ``victims`` is a list of
    ``(key, session)`` pairs for the CALLER to log and close OUTSIDE the
    lock -- closing does I/O (process termination) and must not happen
    while ``_lock`` is held. ``overshoot`` is True if trimming stopped
    early because every remaining candidate was in flight.
    """
    victims = []
    overshoot = False
    while len(_sessions) >= MAX_SESSIONS:
        idle = [k for k in _sessions if _in_flight.get(k, 0) == 0]
        if not idle:
            overshoot = True
            break
        victim_key = min(idle, key = lambda k: _last_used.get(k, 0))
        victims.append((victim_key, _sessions.pop(victim_key)))
        _last_used.pop(victim_key, None)
    return victims, overshoot


def _acquire_core(key, root, factory, start_timeout, *, mark_in_flight):
    """Shared machinery for ``acquire()`` and ``lease()``.

    The ``mark_in_flight`` increment happens inside the same ``_lock``
    section that publishes the session to ``_sessions`` -- deliberately
    NOT after a plain ``acquire()`` call returns. If ``lease()`` instead
    called ``acquire()`` and incremented afterward, the gap between those
    two steps would reopen exactly the race this exists to close: a
    concurrent eviction on another thread would still see ``in_flight == 0``
    during that gap and could pick this session as its victim before the
    increment ever lands.
    """
    with _key_lock(key):
        with _lock:
            existing = _sessions.get(key)
        if existing is not None:
            if existing.is_healthy():
                with _lock:
                    _last_used[key] = time.monotonic()
                    if mark_in_flight:
                        _in_flight[key] = _in_flight.get(key, 0) + 1
                return existing
            with _lock:
                _sessions.pop(key, None)
                _last_used.pop(key, None)
            _close_quietly(existing)

        session = factory(os.path.abspath(root))
        session.start(timeout = start_timeout)   # SessionStartFailed propagates
        with _lock:
            victims, overshoot = _evict_locked()
            _sessions[key] = session
            _last_used[key] = time.monotonic()
            if mark_in_flight:
                _in_flight[key] = _in_flight.get(key, 0) + 1
            live_count = len(_sessions)
    # Logging and closing happen outside _lock, and each logger call is
    # individually guarded: a broken logger (see _atexit_cleanup) must
    # never stop a victim from being closed, and must never propagate out
    # of this function -- for lease() in particular, raising here would
    # skip the caller's try/finally entirely and leak the in_flight
    # increment made above forever, which is worse than the bug this
    # module fixes.
    if overshoot:
        try:
            logger.warning(
                "assist_code: %d language servers live, at MAX_SESSIONS=%d, "
                "but every one is in flight -- not evicting a live request",
                live_count, MAX_SESSIONS,
            )
        except Exception:
            pass
    for victim_key, victim in victims:
        try:
            logger.info("assist_code: evicting idle language server for %s", victim_key)
        except Exception:
            pass
        _close_quietly(victim)
    return session


def acquire(language, root, factory, *, start_timeout = 60.0):
    """Return a live session for (language, root), creating one if needed.

    A dead session is replaced rather than handed back. A server that fails to
    start is NOT cached -- caching a corpse turns one bad start into a
    permanent failure for that workspace.

    Does not protect the returned session from eviction or idle-reaping
    while the caller is using it -- for a session held across an actual
    request, use ``lease()`` instead.
    """
    key = _key(language, root)
    return _acquire_core(key, root, factory, start_timeout, mark_in_flight = False)


@contextlib.contextmanager
def lease(language, root, factory, *, start_timeout = 60.0):
    """Hold a session for the duration of a request, immune to eviction and
    idle-reaping for as long as the ``with`` block is open.

    Context-manager only, deliberately: the decrement runs in a ``finally``,
    so it cannot be skipped by an exception or by a caller that forgets to
    call a release function. A bare increment/decrement pair in the public
    API would make "forgot to release" turn a session permanently
    un-evictable -- a worse failure than the mid-request eviction this
    exists to fix.

    ``_in_flight`` is keyed by the pool key, not by session identity. Each
    ``lease()`` call's own increment and decrement are always a matched
    pair for the same key regardless of what any other thread does to that
    key in between (including replacing a dead session with a fresh one),
    so the count can never under-count while a lease is still open -- the
    one failure direction that would matter, since it's what would let a
    live request be evicted. It can only ever transiently over-count (e.g.
    across a session replacement racing a still-open lease on the old one),
    which just means the pool stays oversized a little longer -- the
    explicitly preferred failure direction, not a dangerous one.
    """
    key = _key(language, root)
    session = _acquire_core(key, root, factory, start_timeout, mark_in_flight = True)
    try:
        yield session
    finally:
        with _lock:
            remaining = _in_flight.get(key, 0) - 1
            if remaining <= 0:
                _in_flight.pop(key, None)
            else:
                _in_flight[key] = remaining


def reap_idle():
    now = time.monotonic()
    stale = []
    with _lock:
        for key, used in list(_last_used.items()):
            if _in_flight.get(key, 0) > 0:
                continue  # a lease is active; never reap out from under it
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
    """Close every session unconditionally, including in-flight ones.

    This is process teardown, not the LRU/idle paths above: there is no
    "later" for an in-flight request to finish into once the process is
    going down, so the in_flight guard that protects acquire()/reap_idle()
    does not apply here.
    """
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
        _last_used.clear()
        _in_flight.clear()
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
