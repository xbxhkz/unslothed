# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The roles API.

Auth is asserted by overriding the dependency, never by monkeypatching the
function: Depends() captures the callable at import time, so a monkeypatch would
leave the real dependency in place and the test would prove nothing.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.inference import model_roles
from core.inference.model_roles import storage
from auth.authentication import get_current_subject
from routes import model_roles as roles_route


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    saved: dict = {}
    monkeypatch.setattr(storage, "get_role_bindings", lambda: dict(saved))
    monkeypatch.setattr(storage, "set_role_bindings", lambda raw: saved.update(raw) or dict(saved))
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: None)
    app = FastAPI()
    app.include_router(roles_route.router, prefix = "/api/model-roles")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def test_get_reports_bindings_and_availability(client):
    client.put("/api/model-roles", json = {"roles": {"coding": {"model": "m"}}})
    body = client.get("/api/model-roles").json()
    assert body["roles"]["coding"]["model"] == "m"
    assert body["roles"]["coding"]["state"] == "missing", "not downloaded in this fixture"
    assert "primary" in body["defaults"]


def test_put_replaces_bindings(client):
    assert client.put("/api/model-roles", json = {"roles": {"coding": {"model": "m"}}}).status_code == 200
    assert client.get("/api/model-roles").json()["roles"]["coding"]["model"] == "m"


def test_put_rejects_a_binding_without_a_model(client):
    response = client.put("/api/model-roles", json = {"roles": {"coding": {}}})
    assert response.status_code == 400
    assert "model" in response.json()["detail"].lower()


def test_put_rejects_a_non_object_roles_payload(client):
    assert client.put("/api/model-roles", json = {"roles": []}).status_code == 400


def test_both_endpoints_require_authentication(monkeypatch, tmp_path):
    """Control: with no dependency override the real auth dependency runs, and an
    unauthenticated request must not reach the handler."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(roles_route.router, prefix = "/api/model-roles")
    unauthenticated = TestClient(app)
    assert unauthenticated.get("/api/model-roles").status_code in (401, 403)
    assert unauthenticated.put("/api/model-roles", json = {"roles": {}}).status_code in (401, 403)
