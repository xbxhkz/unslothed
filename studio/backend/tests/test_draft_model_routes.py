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
        target = _gguf(tmp_path / "m" / "target.gguf")
        _gguf(tmp_path / "m" / "draft.gguf")
        r = client.get("/api/draft-model/candidates", params = {"model_path": str(target)})
        assert r.status_code == 200
        refs = [c["ref"] for c in r.json()["candidates"]]
        assert any("draft.gguf" in x for x in refs)
        assert not any("target.gguf" in x for x in refs), (
            "a model must never be offered as its own drafter"
        )


class TestSelect:
    def test_a_valid_choice_returns_composed_args(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf", 32)
        draft = _gguf(tmp_path / "m" / "draft.gguf", 32)
        r = client.post("/api/draft-model/select", json = {
            "model_path": str(target),
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
            "model_path": str(target),
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
            "model_path": str(target),
            "existing_args": ["--threads", "8", "-md", "/old.gguf"],
            "choice": None,
        })
        assert r.json()["llama_extra_args"] == ["--threads", "8"]
