# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys

import pytest

from core.inference.assist_code import navigation as nav, session as sess

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


@pytest.fixture
def ts_file(tmp_path):
    f = tmp_path / "a.ts"
    f.write_text("const alpha = 1\nconst beta = 2\n\nconsole.log(alpha)\n")
    return f


@pytest.fixture
def started(tmp_path):
    s = sess.Session([sys.executable, FAKE, "normal"], str(tmp_path), language = "typescript")
    s.start(timeout = 10)
    yield s
    s.close()


class TestResolvePosition:
    def test_explicit_line_and_column_convert_to_zero_based(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file), line = 4, column = 13)
        assert err is None
        assert pos == {"line": 3, "character": 12}

    def test_a_symbol_name_is_found_in_the_file(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file), symbol = "beta")
        assert err is None
        assert pos["line"] == 1
        assert pos["character"] == 6

    def test_a_symbol_that_is_absent_says_so(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file), symbol = "nowhere")
        assert pos is None
        assert "nowhere" in err

    def test_neither_symbol_nor_position_is_an_error(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file))
        assert pos is None
        assert "symbol" in err.lower()


class TestOperations:
    def test_definition_returns_one_based_locations(self, started, ts_file):
        locs = nav.definition(started, str(ts_file), {"line": 3, "character": 12}, timeout = 10)
        assert locs == [{"path": os.path.abspath(str(ts_file)), "line": 1, "column": 7}]

    def test_references_returns_every_usage(self, started, ts_file):
        locs = nav.references(started, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        assert len(locs) == 2
        assert locs[0]["line"] == 1 and locs[1]["line"] == 4

    def test_hover_returns_plain_text_without_markdown_fences(self, started, ts_file):
        text = nav.hover(started, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        assert "const a: number" in text
        assert "```" not in text

    def test_symbols_finds_matches(self, started, ts_file):
        got = nav.symbols(started, "alpha", timeout = 10)
        assert got and got[0]["name"] == "alpha"

    def test_no_symbol_matches_is_an_empty_list_not_an_error(self, started, ts_file):
        assert nav.symbols(started, "nothingmatches", timeout = 10) == []
