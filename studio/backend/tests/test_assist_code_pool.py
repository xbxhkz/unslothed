# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys
import threading

import pytest

from core.inference.assist_code import pool, session as sess

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _factory(tmp_path, mode = "normal"):
    def make(root):
        return sess.Session([sys.executable, FAKE, mode], root, language = "typescript")
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
