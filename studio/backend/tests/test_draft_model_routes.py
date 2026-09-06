# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP contract for the draft-model chooser."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.draft_model import router

_GGUF_MAGIC = 0x46554747


def _gguf(path: Path, n_tokens: int = 32) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    key = b"tokenizer.ggml.tokens"
    body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", n_tokens)
    for i in range(n_tokens):
        tok = f"t{i}".encode()
        body += struct.pack("<Q", len(tok)) + tok
    path.write_bytes(body)
    return path


@pytest.fixture
def client():
    """Auth is replaced through dependency_overrides, NOT monkeypatch.

    `Depends(get_current_subject)` captures the function object when the route
    is defined at import time, so reassigning the module attribute afterwards
    changes nothing and every request would still hit real auth. FastAPI's
    override map is keyed on that captured object, which is why it works.
    """
    from auth.authentication import get_current_subject

    app = FastAPI()
    app.include_router(router, prefix = "/api/draft-model")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    return TestClient(app)


class TestCandidates:
    def test_colocated_ggufs_are_offered_and_the_target_itself_is_not(self, tmp_path, client):
        # A genuine local path, still resolved exactly as before this fix round
        # (the `is_local_path` branch of `_resolve_model_path` is a straight
        # `_resolve_real`, unchanged from what this route did previously).
        target = _gguf(tmp_path / "m" / "target.gguf")
        _gguf(tmp_path / "m" / "draft.gguf")
        r = client.get("/api/draft-model/candidates", params = {"model_id": str(target)})
        assert r.status_code == 200
        body = r.json()
        assert body["resolved"] is True
        refs = [c["ref"] for c in body["candidates"]]
        assert any("draft.gguf" in x for x in refs)
        assert not any("target.gguf" in x for x in refs), (
            "a model must never be offered as its own drafter"
        )

    def test_an_unresolvable_repo_id_reports_unresolved_not_empty(self, client):
        # A bare HF-shaped identifier this machine has never cached anything for.
        # Before the fix this was treated as a literal filesystem path (globbing
        # a nonsense directory); it must now come back as "could not resolve",
        # never as a plausible-but-wrong empty candidate list.
        r = client.get(
            "/api/draft-model/candidates",
            params = {
                "model_id": "unsloth-test-fixture-org/does-not-exist-Qwen-GGUF",
                "gguf_variant": "Q4_K_M",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["resolved"] is False
        assert body["candidates"] == []

    def test_a_repo_id_with_no_variant_is_unresolved_not_empty(self, client):
        # cached_gguf_for_load refuses to guess a quant, by design; that must
        # surface the same way as any other unresolvable identifier.
        r = client.get(
            "/api/draft-model/candidates",
            params = {"model_id": "unsloth/Qwen3-4B-GGUF"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["resolved"] is False
        assert body["candidates"] == []


class TestSelect:
    def test_a_valid_choice_returns_composed_args(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf", 32)
        draft = _gguf(tmp_path / "m" / "draft.gguf", 32)
        r = client.post("/api/draft-model/select", json = {
            "model_id": str(target),
            "existing_args": ["--threads", "8"],
            "choice": {"kind": "local", "ref": str(draft)},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["llama_extra_args"] == ["--threads", "8", "--model-draft", str(draft)]

    def test_an_invalid_choice_returns_the_reason_and_no_args(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf", 32)
        draft = _gguf(tmp_path / "m" / "draft.gguf", 64)
        r = client.post("/api/draft-model/select", json = {
            "model_id": str(target),
            "existing_args": ["--threads", "8"],
            "choice": {"kind": "local", "ref": str(draft)},
        })
        assert r.status_code == 200, "a rejected choice is a verdict, not a server error"
        body = r.json()
        assert body["ok"] is False
        assert body["reason"] == "vocab_mismatch"
        assert body["llama_extra_args"] is None, (
            "a rejected choice must not hand back args the UI might save anyway"
        )

    def test_clearing_removes_the_drafter_flags(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf")
        r = client.post("/api/draft-model/select", json = {
            "model_id": str(target),
            "existing_args": ["--threads", "8", "-md", "/old.gguf"],
            "choice": None,
        })
        assert r.json()["llama_extra_args"] == ["--threads", "8"]

    def test_an_unresolvable_repo_id_yields_unresolved_not_no_target(self, client):
        # Given (non-empty model_id) but unresolvable -- must be VERDICT_UNRESOLVED,
        # not VERDICT_NO_TARGET (which means no model_id was given at all) and not
        # a silent pass-through that lets validate_choice treat the raw id as a path.
        r = client.post("/api/draft-model/select", json = {
            "model_id": "unsloth-test-fixture-org/does-not-exist-Qwen-GGUF",
            "gguf_variant": "Q4_K_M",
            "existing_args": [],
            "choice": {"kind": "local", "ref": "draft.gguf"},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["reason"] == "unresolved"
        assert body["llama_extra_args"] is None

    def test_an_hf_choice_does_not_need_the_target_to_resolve(self, client):
        # validate_choice never looks at target_path for an "hf" pick, so an
        # unresolvable target model must not block a perfectly valid hf drafter.
        r = client.post("/api/draft-model/select", json = {
            "model_id": "unsloth-test-fixture-org/does-not-exist-Qwen-GGUF",
            "gguf_variant": "Q4_K_M",
            "existing_args": [],
            "choice": {"kind": "hf", "ref": "unsloth/Qwen3-0.6B-GGUF"},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["llama_extra_args"] == [
            "--spec-draft-hf", "unsloth/Qwen3-0.6B-GGUF",
        ]

    def test_an_accepted_hf_choice_travels_with_the_unverified_reason(self, client):
        # The pin is written, but nothing about the repo was checked. The UI
        # needs to be able to SAY that, so an affirmative ok must not arrive
        # carrying the same reason a fully validated local pick does.
        r = client.post("/api/draft-model/select", json = {
            "existing_args": [],
            "choice": {"kind": "hf", "ref": "unsloth/Qwen3-0.6B-GGUF"},
        })
        body = r.json()
        assert body["ok"] is True
        assert body["reason"] == "unverified"
        assert body["detail"], "an unverified pin with no explanation is a silent pass"

    def test_a_malformed_hf_repo_id_is_rejected_over_http(self, client):
        r = client.post("/api/draft-model/select", json = {
            "existing_args": [],
            "choice": {"kind": "hf", "ref": "/etc/passwd"},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["reason"] == "invalid_repo_id"
        assert body["llama_extra_args"] is None


class TestCurrent:
    """The read-back endpoint. It exists so the frontend never parses the flag
    vocabulary itself -- a picker that did recognised two spellings out of
    seven and showed "Automatic" for five kinds of pinned model."""

    def test_a_pinned_local_drafter_is_reported(self, client):
        r = client.post("/api/draft-model/current", json = {
            "existing_args": ["--threads", "8", "--model-draft", "/m/d.gguf"],
        })
        assert r.status_code == 200
        assert r.json()["pin"] == {"kind": "local", "ref": "/m/d.gguf"}

    def test_a_pinned_hf_drafter_is_reported(self, client):
        r = client.post("/api/draft-model/current", json = {
            "existing_args": ["-hfrd", "org/drafter"],
        })
        assert r.json()["pin"] == {"kind": "hf", "ref": "org/drafter"}

    def test_the_rare_spellings_are_reported_too(self, client):
        # The five the frontend's own mini-parser missed.
        for args, expected in [
            (["-md", "/d.gguf"], {"kind": "local", "ref": "/d.gguf"}),
            (["--spec-draft-model", "/d.gguf"], {"kind": "local", "ref": "/d.gguf"}),
            (["-hfd", "a/b"], {"kind": "hf", "ref": "a/b"}),
            (["--hf-repo-draft", "a/b"], {"kind": "hf", "ref": "a/b"}),
            (["--model-draft=/d.gguf"], {"kind": "local", "ref": "/d.gguf"}),
            (["--model_draft", "/d.gguf"], {"kind": "local", "ref": "/d.gguf"}),
        ]:
            r = client.post("/api/draft-model/current", json = {"existing_args": args})
            assert r.json()["pin"] == expected, args

    def test_no_pin_is_null_not_an_empty_choice(self, client):
        r = client.post("/api/draft-model/current", json = {
            "existing_args": ["--threads", "8"],
        })
        assert r.json()["pin"] is None

    def test_a_round_trip_through_select_reads_back_what_was_pinned(self, tmp_path, client):
        """Binds the write side to the read side over HTTP: whatever /select
        composes, /current must report as the pin. Two endpoints that disagree
        here is the two-parser failure moved one layer out."""
        target = _gguf(tmp_path / "m" / "target.gguf", 32)
        draft = _gguf(tmp_path / "m" / "draft.gguf", 32)
        composed = client.post("/api/draft-model/select", json = {
            "model_id": str(target),
            "existing_args": ["--threads", "8"],
            "choice": {"kind": "local", "ref": str(draft)},
        }).json()["llama_extra_args"]
        r = client.post("/api/draft-model/current", json = {"existing_args": composed})
        assert r.json()["pin"] == {"kind": "local", "ref": str(draft)}
