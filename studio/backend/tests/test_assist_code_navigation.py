# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys

import pytest

from core.inference.assist_code import jsonrpc, navigation as nav, session as sess

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

    def test_a_fractional_line_is_rejected_not_silently_truncated(self, started, ts_file):
        """int(4.7) succeeds and truncates to 4 -- accepting a fractional
        line as if it were exact would contradict the "must be whole
        numbers" error text every other malformed input gets."""
        pos, err = nav.resolve_position(started, str(ts_file), line = 4.7, column = 1)
        assert pos is None
        assert "whole number" in err.lower()

    def test_a_whole_valued_float_line_is_still_accepted(self, started, ts_file):
        """4.0 IS a whole number -- only a nonzero fractional part is
        rejected."""
        pos, err = nav.resolve_position(started, str(ts_file), line = 4.0, column = 1.0)
        assert err is None
        assert pos == {"line": 3, "character": 0}


class TestOperations:
    def test_definition_returns_one_based_locations(self, started, ts_file):
        locs = nav.definition(started, str(ts_file), {"line": 3, "character": 12}, timeout = 10)
        assert locs == [{"path": os.path.abspath(str(ts_file)), "line": 1, "column": 7}]

    def test_references_returns_every_usage(self, started, ts_file):
        locs = nav.references(started, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        assert len(locs) == 2
        assert locs[0]["line"] == 1 and locs[0]["column"] == 7
        assert locs[1]["line"] == 4 and locs[1]["column"] == 3

    def test_hover_returns_plain_text_without_markdown_fences(self, started, ts_file):
        text = nav.hover(started, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        assert "const a: number" in text
        assert "```" not in text

    def test_symbols_finds_matches(self, started, ts_file):
        got = nav.symbols(started, "alpha", timeout = 10)
        assert got
        assert got[0]["name"] == "alpha"
        assert got[0]["line"] == 1
        assert got[0]["column"] == 7
        # The fake server's workspace/symbol location URI is a hardcoded,
        # session-independent "file:///ws/a.ts" (Task 7 round-1 review,
        # Minor 5 -- assessed as a narrow gap, not fixed). Asserting the
        # known constant still pins _loc()'s URI-to-path mapping for the
        # nested "location" shape, same as definition/references already
        # cover for the flat shape.
        assert got[0]["path"] == sess.uri_to_path("file:///ws/a.ts")

    def test_no_symbol_matches_is_an_empty_list_not_an_error(self, started, ts_file):
        assert nav.symbols(started, "nothingmatches", timeout = 10) == []


class TestReferencesError:
    def test_a_genuine_protocol_error_is_not_swallowed_as_no_references(self, tmp_path, ts_file):
        """A server that answers textDocument/references with a JSON-RPC
        error object is alive and rejected the request -- that is not the
        same situation as "nothing calls this." An empty list here is
        action-guiding (it reads as "safe to delete"), so a real failure
        must never be allowed to look like a clean answer. See
        navigation.py's module docstring."""
        s = sess.Session([sys.executable, FAKE, "refs_error"], str(tmp_path), language = "typescript")
        s.start(timeout = 10)
        try:
            with pytest.raises(jsonrpc.LspError):
                nav.references(s, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        finally:
            s.close()


class TestDefinitionError:
    def test_a_genuine_protocol_error_is_not_swallowed_as_no_definition(self, tmp_path, ts_file):
        """Same failure mode as TestReferencesError, for definition(). Task
        7 round-1 review: a single shared negative control (references only)
        left definition(), hover(), and symbols() each able to regress back
        to swallowing LspError with nothing to catch it -- each function
        needs its own fake-server error mode and its own test."""
        s = sess.Session([sys.executable, FAKE, "def_error"], str(tmp_path), language = "typescript")
        s.start(timeout = 10)
        try:
            with pytest.raises(jsonrpc.LspError):
                nav.definition(s, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        finally:
            s.close()


class TestHoverError:
    def test_a_genuine_protocol_error_is_not_swallowed_as_empty_hover(self, tmp_path, ts_file):
        """Same failure mode as TestReferencesError, for hover()."""
        s = sess.Session([sys.executable, FAKE, "hover_error"], str(tmp_path), language = "typescript")
        s.start(timeout = 10)
        try:
            with pytest.raises(jsonrpc.LspError):
                nav.hover(s, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        finally:
            s.close()


class TestSymbolsError:
    def test_a_genuine_protocol_error_is_not_swallowed_as_no_matches(self, tmp_path):
        """Same failure mode as TestReferencesError, for symbols(). No
        ts_file/open_document involved -- workspace/symbol is not scoped to
        a single document."""
        s = sess.Session([sys.executable, FAKE, "symbols_error"], str(tmp_path), language = "typescript")
        s.start(timeout = 10)
        try:
            with pytest.raises(jsonrpc.LspError):
                nav.symbols(s, "alpha", timeout = 10)
        finally:
            s.close()


class TestHoverContentShapes:
    """LSP hover ``contents`` may be a plain string, a ``{language, value}``
    pair, a ``MarkupContent`` (``{kind, value}``), or a list mixing any of
    those. The shipped suite's only end-to-end hover test
    (``test_hover_returns_plain_text_without_markdown_fences``) exercises
    just the MarkupContent shape through a live fake-server round trip --
    a regression in any of the other three branches of ``_hover_text``
    would be invisible to it. These test the result-shaping helper
    directly (no live session needed, since it's a pure function of the
    raw LSP result) to cover the rest."""

    def test_a_plain_string(self):
        assert nav._hover_from_result({"contents": "just plain text"}) == "just plain text"

    def test_a_language_value_pair(self):
        """The deprecated-but-still-valid `{language, value}` MarkedString
        shape -- distinct from MarkupContent's `{kind, value}` only in the
        name of the sibling key, which _hover_text ignores either way."""
        result = {"contents": {"language": "typescript", "value": "const a: number"}}
        assert nav._hover_from_result(result) == "const a: number"

    def test_a_list_mixing_shapes_with_adversarial_fences(self):
        """A list of three parts: a string with a ```py-tagged fence block,
        a {language, value} pair with no fence, and a string with an
        untagged fence block followed by a second, trailing fence
        immediately butted up against text with no separating newline.

        Current (pinned, not asserted-ideal) behaviour: the fence-stripping
        regex's language-tag group (``[a-zA-Z0-9_+-]*``) is greedy and has
        no way to distinguish a real language tag from the start of literal
        content immediately following a fence marker -- so the trailing
        "```trailing" is consumed whole, silently dropping "trailing" from
        the output along with the fence. Real MarkupContent from language
        servers puts fences on their own line, so this is a narrow, known
        quirk (not in this round's fix list) rather than a live risk --
        pinned here so a future change to the stripping regex is a
        deliberate decision, not an unnoticed behaviour change.
        """
        contents = [
            "```py\ndef f():\n    pass\n```",
            {"language": "ts", "value": "const a: number"},
            "```\nplain fenced\n```trailing",
        ]
        cleaned = nav._hover_from_result({"contents": contents})
        assert cleaned == "def f():\n    pass\nconst a: number\nplain fenced"
        assert "```" not in cleaned
