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
