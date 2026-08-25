# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys

import pytest

from core.inference.assist_code import diagnostics, session as sess

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _started(tmp_path, mode):
    s = sess.Session([sys.executable, FAKE, mode], str(tmp_path), language = "typescript")
    s.start(timeout = 10)
    return s


@pytest.fixture
def ts_file(tmp_path):
    f = tmp_path / "a.ts"
    f.write_text("const a = 1\nconst b = 2\nconst c: number = 'x'\nlet d\nlet x\n")
    return f


class TestPushPath:
    def test_pushed_diagnostics_are_collected(self, tmp_path, ts_file):
        s = _started(tmp_path, "normal")
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 10)
            assert len(got) == 1
            assert got[0]["severity"] == "error"
            assert "not assignable" in got[0]["message"]
        finally:
            s.close()

    def test_line_and_column_are_converted_to_one_based(self, tmp_path, ts_file):
        """LSP is 0-based; humans and every compiler are 1-based. Off-by-one
        here sends the model to the wrong line."""
        s = _started(tmp_path, "normal")
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 10)
            assert got[0]["line"] == 3      # LSP line 2
            assert got[0]["column"] == 7    # LSP character 6
        finally:
            s.close()


class TestPullPath:
    def test_pull_is_used_when_the_server_advertises_it(self, tmp_path, ts_file):
        s = _started(tmp_path, "pull")
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 10)
            assert len(got) == 1
            assert got[0]["severity"] == "warning"
            assert "never used" in got[0]["message"]
            assert got[0]["line"] == 5
        finally:
            s.close()


class TestNoResults:
    def test_a_server_that_publishes_nothing_yields_an_empty_list_not_an_error(
        self, tmp_path, ts_file
    ):
        """A clean file is a valid answer, not a failure."""
        s = _started(tmp_path, "wedged")   # never pushes, never answers
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 1)
            assert got == []
        finally:
            s.close()
