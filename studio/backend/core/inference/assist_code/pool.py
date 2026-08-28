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

Idle reaping RUNS. ``reap_idle()`` existed, was documented and was tested,
and was called by nothing but its own tests -- so ``IDLE_TIMEOUT_SECONDS``
never once fired outside the suite, and the argument two paragraphs up
about 200MB-1GB per server was being made by a module that never reclaimed
any of it. A lazily-started daemon thread (``_ensure_reaper_started_locked``,
after ``mcp_client._stdio_session_reaper`` :813) now drives it. Lazy so a
process that never uses a code tool never carries the thread.
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
# How often the background reaper wakes. Mirrors mcp_client's
# _STDIO_SESSION_REAP_INTERVAL (:292). Read fresh on each iteration so it can
# be lowered in tests without restarting the thread.
REAP_INTERVAL_SECONDS = 30.0

_lock = threading.Lock()
_key_locks = {}
_sessions = {}      # key -> session
_last_used = {}     # key -> monotonic timestamp
_in_flight = {}      # key -> active-lease count; guarded by _lock, see lease()
_reaper_started = False


def _key(language, root):
    return (language, os.path.abspath(root))


class _KeyLock:
    """A per-key lock plus a count of who is currently interested in it.

    The count is the whole point, and it is why this is a class rather than
    the bare ``threading.Lock`` this module used to store. Without it there
    is no safe moment to remove an entry from ``_key_locks``: dropping one
    while another thread is blocked on it hands the next arrival a DIFFERENT
    lock object for the same key, which is exactly the "two threads both
    spawn a server for the same workspace" race the per-key lock exists to
    prevent. So the old code never removed one, and ``_key_locks`` grew
    without bound -- one permanent entry per (language, root) ever seen.

    Follows ``mcp_client._StdioKeyLock`` (:540) and its
    borrow/return/discard trio (:599-:613), which pool.py already cites as
    its model and had adopted everything else from.
    """

    __slots__ = ("lock", "users")

    def __init__(self):
        self.lock = threading.Lock()
        self.users = 0


def _borrow_key_lock(key):
    """Return a stable per-key lock, marking the caller as interested."""
    with _lock:
        key_lock = _key_locks.get(key)
        if key_lock is None:
            key_lock = _KeyLock()
            _key_locks[key] = key_lock
        key_lock.users += 1
        return key_lock


def _discard_key_lock_locked(key):
    """Caller holds ``_lock``. Drop the key's lock if nothing needs it.

    Two conditions, both required: nobody is currently holding or waiting on
    it (``users == 0``), and no session exists under the key (a live session
    will be acquired again, and re-creating its lock in between would let two
    threads race for it). Mirrors ``mcp_client._discard_stdio_key_lock``
    (:606).
    """
    key_lock = _key_locks.get(key)
    if key_lock is not None and key_lock.users == 0 and key not in _sessions:
        _key_locks.pop(key, None)


def _return_key_lock(key, key_lock):
    with _lock:
        key_lock.users -= 1
        _discard_key_lock_locked(key)


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
        # An evicted key may now be droppable. Without this, eviction was a
        # second path that left a key lock behind forever -- caught by
        # test_key_locks_do_not_grow_without_bound, which saw 6 locks for 4
        # sessions after the pool had cycled through six workspaces.
        _discard_key_lock_locked(victim_key)
    return victims, overshoot


def _acquire_core(key, root, factory, start_timeout, *, mark_in_flight):
    """Shared machinery for ``acquire()`` and ``lease()``.

    The ``mark_in_flight`` increment happens inside the same ``_lock``
    section that publishes/re-validates the session -- for BOTH the
    new-session path below and the existing-session path just above it.

    Fix round 2: the existing-session path used to fetch ``existing``,
    release ``_lock``, call ``existing.is_healthy()`` OUTSIDE the lock, and
    only THEN re-acquire ``_lock`` to claim it -- the exact two-phase
    "observe, then claim" shape this docstring already warned the
    new-session path against, just in the one place that still had it.
    Reproduced deterministically under review: pausing inside
    ``is_healthy()`` right after it computes a genuinely healthy result
    but before returning, a concurrent eviction for a DIFFERENT key could
    land in that gap -- ``in_flight`` for this key was still 0, so nothing
    protected it -- evict and close THIS session, and this function would
    then increment ``in_flight`` and hand back an already-closed session,
    believing it protected. ``is_healthy()`` is just a ``poll()`` plus a
    flag read (no I/O), so folding it into the same locked section that
    was already doing the claim is cheap and closes the gap completely: a
    concurrent eviction on another thread now either fully precedes this
    whole observe-and-claim (sees the session, in_flight still 0, evicts
    it -- correctly, nobody had claimed it yet) or fully follows it (sees
    in_flight == 1, correctly skips it). It can no longer land in between.
    """
    key_lock = _borrow_key_lock(key)
    try:
        with key_lock.lock:
            stale = None
            with _lock:
                existing = _sessions.get(key)
                if existing is not None:
                    if existing.is_healthy():
                        _last_used[key] = time.monotonic()
                        if mark_in_flight:
                            _in_flight[key] = _in_flight.get(key, 0) + 1
                        return existing
                    stale = _sessions.pop(key, None)
                    _last_used.pop(key, None)
            if stale is not None:
                _close_quietly(stale)

            session = factory(os.path.abspath(root))
            session.start(timeout = start_timeout)   # SessionStartFailed propagates
            with _lock:
                victims, overshoot = _evict_locked()
                _sessions[key] = session
                _last_used[key] = time.monotonic()
                if mark_in_flight:
                    _in_flight[key] = _in_flight.get(key, 0) + 1
                live_count = len(_sessions)
                # Started here, not at import: a process that never uses a
                # code tool should not carry a thread for one. Mirrors
                # mcp_client (:682), which likewise starts its reaper in the
                # locked section that publishes a session.
                _ensure_reaper_started_locked()
    finally:
        _return_key_lock(key, key_lock)
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

    Also re-enforces MAX_SESSIONS on release, synchronously, in this same
    ``finally`` -- not deferred to a background thread or to whenever the
    next brand-new key happens to be acquired. ``_evict_locked`` only trims
    idle sessions at insert time, so the pool can be left transiently over
    cap while every cached session was busy (its own docstring); without
    reclaiming here, that overshoot would otherwise persist indefinitely if
    no new key is ever acquired afterward, or sit until the idle reaper's
    much longer ``IDLE_TIMEOUT_SECONDS`` window elapses. Language servers
    are 200MB-1GB each, so an over-cap pool held idle is real memory, not
    just a bookkeeping nicety. Mirrors ``mcp_client._release_stdio_session``
    (:701), which the original brief for round 1's fix omitted -- this
    round's gap, not a prior-round oversight.
    """
    key = _key(language, root)
    session = _acquire_core(key, root, factory, start_timeout, mark_in_flight = True)
    try:
        yield session
    finally:
        victims = []
        with _lock:
            remaining = _in_flight.get(key, 0) - 1
            if remaining <= 0:
                _in_flight.pop(key, None)
            else:
                _in_flight[key] = remaining
            _last_used[key] = time.monotonic()
            # Never pick the key we just released as its own reclaim victim
            # (mcp_client's "never evict the session we just used" -- its
            # last_used is freshest, so this is defensive, not load-bearing
            # in the common case, but the new-session path already showed
            # a just-touched entry CAN end up as the only idle candidate).
            while len(_sessions) > MAX_SESSIONS:
                idle = [k for k in _sessions if k != key and _in_flight.get(k, 0) == 0]
                if not idle:
                    break
                victim_key = min(idle, key = lambda k: _last_used.get(k, 0))
                victims.append((victim_key, _sessions.pop(victim_key)))
                _last_used.pop(victim_key, None)
                _discard_key_lock_locked(victim_key)   # see _evict_locked
        for victim_key, victim in victims:
            try:
                logger.info("assist_code: evicting idle language server for %s", victim_key)
            except Exception:
                pass
            _close_quietly(victim)


def _ensure_reaper_started_locked():
    """Caller holds ``_lock``. Start the idle reaper once, lazily.

    ``reap_idle()`` was written, documented and tested, and then never
    called by anything but its own tests -- so ``IDLE_TIMEOUT_SECONDS`` never
    fired in production. After a single code tool call a language server
    stayed resident until a fifth distinct workspace forced an LRU eviction
    or the process exited; at ``MAX_SESSIONS = 4`` that is up to ~4GB held
    indefinitely after one chat message. This module's own docstring makes
    the case that this memory matters ("Language servers are 200MB-1GB
    each") -- it simply never ran the mechanism that reclaims it.

    ``mcp_client._stdio_session_reaper`` (:813) is the missing piece, the
    same file this module already borrowed ``_evict_stdio_lru_locked`` and
    ``_release_stdio_session`` from.

    Starting the thread while holding ``_lock`` is safe and matches
    mcp_client: the new thread's first action is to sleep, so it cannot
    contend for the lock the starter is holding.
    """
    global _reaper_started
    if _reaper_started:
        return
    _reaper_started = True
    try:
        threading.Thread(
            target = _reaper, name = "assist-code-lsp-reaper", daemon = True,
        ).start()
    except Exception:
        # A process that cannot spawn the reaper still has a working pool;
        # it just keeps the pre-existing behaviour of never reaping.
        _reaper_started = False


def _reaper():
    """Drive ``reap_idle()`` forever. Daemon thread; must never raise.

    Daemon so it can never hold the interpreter open, and the loop body is
    wrapped so a single bad iteration cannot kill the thread and silently
    restore the "nothing ever reaps" behaviour this exists to fix.

    ``REAP_INTERVAL_SECONDS`` is read fresh each iteration rather than
    captured, so a test can lower it without restarting the thread.

    The logger call is guarded like every other in this module: during
    interpreter shutdown a daemon thread can still be mid-loop while logging
    handlers' streams are already closed, and a write then raises (see
    ``_atexit_cleanup``).
    """
    while True:
        time.sleep(max(0.05, REAP_INTERVAL_SECONDS))
        try:
            reap_idle()
        except Exception as exc:  # noqa: BLE001 - a bad sweep must not kill the thread
            try:
                logger.debug("assist_code: idle reaper iteration failed: %s", exc)
            except Exception:
                pass


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
        # Now that the sessions are gone, the keys' locks may be droppable --
        # this is the only place besides _return_key_lock where a key stops
        # having a session. Mirrors mcp_client's own reap (:805).
        for key, _session in stale:
            _discard_key_lock_locked(key)
    for key, session in stale:
        if session is not None:
            try:
                logger.info("assist_code: reaping idle language server for %s", key)
            except Exception:
                pass
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

    ``_key_locks`` is cleared here too. It was the one map this function
    missed, so every (language, root) ever seen left a permanent entry behind
    even across a full shutdown -- the unbounded-growth half of the same
    finding that ``_KeyLock``'s user count fixes for the steady-state case.
    Safe at this point for the same reason closing in-flight sessions is:
    this is process teardown, so there is no later arrival to race with.
    """
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
        _last_used.clear()
        _in_flight.clear()
        _key_locks.clear()
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
