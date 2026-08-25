# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys

import pytest

from core.inference.assist_code import session as sess
from core.inference.assist_code import jsonrpc

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _session(tmp_path, mode = "normal"):
    return sess.Session([sys.executable, FAKE, mode], str(tmp_path), language = "typescript")


class TestHandshake:
    def test_start_completes_and_records_capabilities(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            assert s.capabilities.get("definitionProvider") is True
        finally:
            s.close()

    def test_supports_reads_the_negotiated_capabilities(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            assert s.supports("hoverProvider")
            assert not s.supports("renameProvider")
        finally:
            s.close()

    def test_pull_diagnostics_are_detected_only_when_advertised(self, tmp_path):
        push = _session(tmp_path, "normal")
        pull = _session(tmp_path, "pull")
        try:
            push.start(timeout = 10)
            pull.start(timeout = 10)
            assert not push.supports("diagnosticProvider")
            assert pull.supports("diagnosticProvider")
        finally:
            push.close()
            pull.close()

    def test_a_server_that_never_answers_initialize_fails_to_start(self, tmp_path):
        # NOTE: brief said "wedged", corrected to "deaf" -- wedged answers
        # initialize and only then goes silent, so start() would succeed and
        # this test would fail. deaf never answers anything, including the
        # handshake, which is what this test actually needs.
        s = _session(tmp_path, "deaf")
        try:
            with pytest.raises(sess.SessionStartFailed) as e:
                s.start(timeout = 1)
            assert "did not" in str(e.value).lower() or "timed out" in str(e.value).lower()
        finally:
            s.close()

    def test_a_command_that_does_not_exist_fails_to_start_with_a_readable_error(self, tmp_path):
        s = sess.Session(["definitely-not-a-real-binary-xyz"], str(tmp_path), language = "typescript")
        try:
            with pytest.raises(sess.SessionStartFailed) as e:
                s.start(timeout = 5)
            assert "definitely-not-a-real-binary-xyz" in str(e.value)
        finally:
            s.close()

    def test_start_reaps_the_subprocess_when_the_handshake_fails(self, tmp_path):
        # Regression: start() used to raise SessionStartFailed after already
        # spawning a real transport/process, but left the process running.
        # Every other test in this file closes in a `finally`, which makes
        # start()'s own cleanup (or lack of it) invisible -- this test
        # deliberately does NOT call close(), because the failure this
        # guards against is a caller (Task 4's pool `acquire()`) letting
        # SessionStartFailed propagate without ever calling close(). If
        # start() doesn't reap the process itself, this test both fails its
        # assertion AND leaks the subprocess.
        s = _session(tmp_path, "deaf")
        with pytest.raises(sess.SessionStartFailed):
            s.start(timeout = 1)
        assert s._proc is not None
        assert s._proc.poll() is not None


class TestDocuments:
    def test_open_document_sends_the_file_contents(self, tmp_path):
        f = tmp_path / "a.ts"
        f.write_text("const a: number = 'x'\n")
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            s.open_document(str(f))
            note = s.wait_notification(
                "textDocument/publishDiagnostics", lambda p: True, timeout = 10)
            assert note is not None
        finally:
            s.close()

    def test_opening_the_same_document_twice_does_not_resend_didopen(self, tmp_path):
        f = tmp_path / "a.ts"
        f.write_text("x\n")
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            s.open_document(str(f))
            s.open_document(str(f))
            assert s.opened_count == 1
        finally:
            s.close()

    def test_open_document_raises_documentreaderror_not_sessionstartfailed(self, tmp_path):
        # Regression: a missing/unreadable file must not look like a startup
        # failure. Session.request() already keeps post-start failures
        # distinct from SessionStartFailed for the same reason -- a caller
        # doing `except SessionStartFailed: restart_the_session()` would
        # otherwise tear down a perfectly healthy server over one bad file.
        missing = tmp_path / "does_not_exist.ts"
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            with pytest.raises(sess.DocumentReadError):
                s.open_document(str(missing))
        finally:
            s.close()

    def test_open_document_rejects_a_file_over_the_size_cap(self, tmp_path):
        big = tmp_path / "big.ts"
        big.write_bytes(b"x" * (sess._MAX_DOCUMENT_BYTES + 1))
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            with pytest.raises(sess.DocumentReadError):
                s.open_document(str(big))
            assert s.opened_count == 0
        finally:
            s.close()

    def test_wait_notification_raises_when_the_session_was_never_started(self, tmp_path):
        # Consistency: request() already raises LspClosed for this exact
        # condition. wait_notification() silently returning None instead
        # made the same "never started" state look like "no matching
        # notification yet" to a caller.
        s = _session(tmp_path)
        with pytest.raises(jsonrpc.LspClosed):
            s.wait_notification("textDocument/publishDiagnostics", lambda p: True, timeout = 0.1)


class TestHealth:
    def test_a_live_session_is_healthy(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            assert s.is_healthy()
        finally:
            s.close()

    def test_a_closed_session_is_not_healthy(self, tmp_path):
        s = _session(tmp_path)
        s.start(timeout = 10)
        s.close()
        assert not s.is_healthy()

    def test_uri_round_trip_survives_spaces_and_drive_letters(self, tmp_path):
        d = tmp_path / "a dir"
        d.mkdir()
        f = d / "b.ts"
        f.write_text("x")
        uri = sess.path_to_uri(str(f))
        assert uri.startswith("file:///")
        assert sess.uri_to_path(uri) == os.path.abspath(str(f))

    def test_uri_to_path_resolves_the_unc_authority_not_the_current_drive(self):
        # Regression: urlparse puts a UNC host in `netloc`, not `path`.
        # Reading only `.path` silently dropped the host and substituted the
        # current drive, returning a plausible-looking but wrong local path
        # (e.g. "C:\\share\\dir\\file.ts") instead of raising or resolving
        # correctly.
        uri = "file://SERVER/share/dir/file.ts"
        path = sess.uri_to_path(uri)
        assert path == r"\\SERVER\share\dir\file.ts"
        assert sess.path_to_uri(path) == uri
