# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys
import threading
import time

import pytest

from core.inference.assist_code import pool, session as sess

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _factory(tmp_path, mode = "normal"):
    def make(root):
        return sess.Session([sys.executable, FAKE, mode], root, language = "typescript")
    return make


def _slow_factory(tmp_path, delay):
    """Like _factory(tmp_path, "slow") but with a configurable per-request
    delay -- see lsp_fake_server.py's module docstring for why the
    in-flight-eviction regression test needs one longer than
    jsonrpc.Transport.close()'s hardcoded 2.0s graceful-shutdown timeout."""
    def make(root):
        return sess.Session([sys.executable, FAKE, "slow", str(delay)], root, language = "typescript")
    return make


@pytest.fixture(autouse = True)
def _clean_pool():
    pool.shutdown_all()
    yield
    pool.shutdown_all()


class TestReuse:
    def test_the_second_acquire_returns_the_same_session(self, tmp_path):
        a = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        b = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        assert a is b
        assert pool.stats()["live"] == 1

    def test_different_roots_get_different_sessions(self, tmp_path):
        one = tmp_path / "one"; one.mkdir()
        two = tmp_path / "two"; two.mkdir()
        a = pool.acquire("typescript", str(one), _factory(tmp_path))
        b = pool.acquire("typescript", str(two), _factory(tmp_path))
        assert a is not b
        assert pool.stats()["live"] == 2

    def test_different_languages_on_one_root_get_different_sessions(self, tmp_path):
        def make_cs(root):
            return sess.Session([sys.executable, FAKE, "normal"], root, language = "csharp")
        a = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        b = pool.acquire("csharp", str(tmp_path), make_cs)
        assert a is not b

    def test_concurrent_acquires_of_one_key_create_exactly_one_session(self, tmp_path):
        """Without per-key locking this spawns N servers and leaks N-1."""
        seen = []
        def worker():
            seen.append(pool.acquire("typescript", str(tmp_path), _factory(tmp_path)))
        threads = [threading.Thread(target = worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len({id(s) for s in seen}) == 1
        assert pool.stats()["live"] == 1


class TestRecovery:
    def test_a_dead_session_is_replaced(self, tmp_path):
        a = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        a.close()
        b = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        assert b is not a
        assert b.is_healthy()

    def test_a_server_that_cannot_start_reports_rather_than_caching_a_corpse(self, tmp_path):
        def bad(root):
            return sess.Session(["definitely-not-a-real-binary-xyz"], root, language = "typescript")
        with pytest.raises(sess.SessionStartFailed):
            pool.acquire("typescript", str(tmp_path), bad)
        assert pool.stats()["live"] == 0


class TestEviction:
    def test_exceeding_the_cap_evicts_the_least_recently_used(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pool, "MAX_SESSIONS", 2)
        roots = []
        for name in ("a", "b", "c"):
            d = tmp_path / name; d.mkdir(); roots.append(str(d))
        first = pool.acquire("typescript", roots[0], _factory(tmp_path))
        pool.acquire("typescript", roots[1], _factory(tmp_path))
        pool.acquire("typescript", roots[2], _factory(tmp_path))
        assert pool.stats()["live"] == 2
        assert not first.is_healthy(), "the evicted session's process must be closed"

    def test_an_idle_session_is_reaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pool, "IDLE_TIMEOUT_SECONDS", 0)
        s = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        pool.reap_idle()
        assert pool.stats()["live"] == 0
        assert not s.is_healthy()


class TestShutdown:
    def test_shutdown_all_closes_every_session(self, tmp_path):
        one = tmp_path / "one"; one.mkdir()
        two = tmp_path / "two"; two.mkdir()
        a = pool.acquire("typescript", str(one), _factory(tmp_path))
        b = pool.acquire("typescript", str(two), _factory(tmp_path))
        pool.shutdown_all()
        assert pool.stats()["live"] == 0
        assert not a.is_healthy() and not b.is_healthy()

    def test_the_atexit_handler_never_raises_even_with_broken_logging(self, tmp_path, monkeypatch):
        """atexit runs after log streams may be closed. llama_cpp.py:20925
        documents this: a logging call there surfaced as unrelated tracebacks
        printed after the test summary."""
        pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        import logging
        class _Exploding:
            def info(self, *a, **k): raise ValueError("stream closed")
            def warning(self, *a, **k): raise ValueError("stream closed")
            def exception(self, *a, **k): raise ValueError("stream closed")
        monkeypatch.setattr(pool, "logger", _Exploding())
        pool._atexit_cleanup()          # must not raise
        assert logging.raiseExceptions is True, "the flag must be restored"


class _RecordingLogger:
    """Captures calls instead of writing anywhere, so a test can assert a
    warning was logged without depending on structlog/caplog wiring."""
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, *a, **k): self.infos.append((a, k))
    def warning(self, *a, **k): self.warnings.append((a, k))
    def exception(self, *a, **k): pass


class TestInFlightProtection:
    """Fix round 1: eviction/reap picked victims purely by last_used, with
    no notion of "a caller is mid-request against this session". Reproduced
    under review at MAX_SESSIONS=1: acquiring key1, starting a request
    against it, and acquiring key2 while that request was still in flight
    evicted and closed key1 out from under the caller, who then saw an
    unrelated LspClosed instead of its own request completing.
    """

    def test_eviction_skips_an_in_flight_session_and_picks_the_next_lru(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pool, "MAX_SESSIONS", 2)
        roots = []
        for name in ("a", "b", "c"):
            d = tmp_path / name; d.mkdir(); roots.append(str(d))
        a = pool.acquire("typescript", roots[0], _factory(tmp_path))
        key_a = pool._key("typescript", roots[0])
        with pool.lease("typescript", roots[0], _factory(tmp_path)) as leased_a:
            assert leased_a is a
            # Force explicit, deterministic ordering rather than trusting
            # that two back-to-back time.monotonic() calls land apart: `a`
            # must look OLDEST despite the lease's own last_used refresh,
            # so a naive LRU scan (ignoring in_flight) would still pick it.
            pool._last_used[key_a] = 1.0
            b = pool.acquire("typescript", roots[1], _factory(tmp_path))
            key_b = pool._key("typescript", roots[1])
            pool._last_used[key_b] = 2.0
            pool.acquire("typescript", roots[2], _factory(tmp_path))
            assert pool.stats()["live"] == 2
            assert a.is_healthy(), "the in-flight session must not be evicted"
            assert not b.is_healthy(), "eviction must fall through to the next-LRU idle session"

    def test_second_acquire_at_cap_one_does_not_close_the_in_flight_session_and_logs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pool, "MAX_SESSIONS", 1)
        rec = _RecordingLogger()
        monkeypatch.setattr(pool, "logger", rec)
        one = tmp_path / "one"; one.mkdir()
        two = tmp_path / "two"; two.mkdir()
        a = pool.acquire("typescript", str(one), _factory(tmp_path))
        with pool.lease("typescript", str(one), _factory(tmp_path)) as leased_a:
            assert leased_a is a
            b = pool.acquire("typescript", str(two), _factory(tmp_path))
            assert pool.stats()["live"] == 2, "the pool must overshoot the cap rather than evict a live session"
            assert a.is_healthy()
            assert b.is_healthy()
        assert rec.warnings, "the overshoot must be logged"

    def test_reap_idle_skips_an_in_flight_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pool, "IDLE_TIMEOUT_SECONDS", 0)
        a = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        with pool.lease("typescript", str(tmp_path), _factory(tmp_path)) as leased_a:
            assert leased_a is a
            pool.reap_idle()
            assert pool.stats()["live"] == 1, "an in-flight session must survive reap_idle"
            assert a.is_healthy()
        # Once the lease releases, the same session is idle-eligible again.
        pool.reap_idle()
        assert pool.stats()["live"] == 0

    def test_lease_decrements_in_flight_even_on_an_exception(self, tmp_path):
        key = pool._key("typescript", str(tmp_path))
        with pytest.raises(ValueError):
            with pool.lease("typescript", str(tmp_path), _factory(tmp_path)):
                assert pool._in_flight.get(key, 0) == 1
                raise ValueError("boom")
        assert pool._in_flight.get(key, 0) == 0, "the decrement must run even though the block raised"

    def test_reviewer_repro_an_in_flight_request_survives_a_concurrent_acquire_at_cap(self, tmp_path, monkeypatch):
        """The reviewer's exact reproduction. Before the in_flight guard,
        this made the worker's request fail with LspClosed("server exited
        while handling workspace/symbol") because the concurrent acquire
        below evicted and closed its session mid-request.

        Uses a 3s per-request delay, not the fake server's default 0.8s:
        Transport.close() (jsonrpc.py:245) tries a graceful "shutdown"
        request with a hardcoded 2.0s timeout before it ever force-kills.
        The fake server processes messages strictly FIFO on one thread, so
        at the default 0.8s delay, close()'s shutdown message -- queued
        behind the pending workspace/symbol request -- never even gets read
        until after that request's real reply is already written: closing
        mid-request would lose that race and answer normally regardless of
        whether eviction was supposed to protect the session, which made an
        earlier version of this test's own negative control inert. A delay
        past 2.0s makes the graceful shutdown time out and fall through to
        an abrupt kill while the request is still genuinely unanswered,
        which is what actually distinguishes protected from unprotected.
        """
        monkeypatch.setattr(pool, "MAX_SESSIONS", 1)
        one = tmp_path / "one"; one.mkdir()
        two = tmp_path / "two"; two.mkdir()

        result = {}
        started = threading.Event()

        def worker():
            with pool.lease("typescript", str(one), _slow_factory(tmp_path, delay = 3.0)) as s:
                started.set()
                try:
                    result["value"] = s.request("workspace/symbol", {"query": ""}, timeout = 8)
                except Exception as e:
                    result["error"] = e

        t = threading.Thread(target = worker)
        t.start()
        assert started.wait(timeout = 5), "worker never entered its lease"
        time.sleep(0.1)  # let the request actually reach the wire before racing it
        pool.acquire("typescript", str(two), _factory(tmp_path))
        t.join(timeout = 10)
        assert not t.is_alive(), "worker thread never finished"
        assert "error" not in result, f"in-flight request was killed: {result.get('error')!r}"
        assert "value" in result, "in-flight request never completed"
