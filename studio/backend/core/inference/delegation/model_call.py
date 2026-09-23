# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One turn from the loaded backend, as an OpenAI-style assistant message.

llama_cpp is do-not-edit and already exposes base_url and _auth_headers, so this
talks to the server it is already running rather than starting anything.

Nothing here checks WHICH model answers: that check belongs to the caller, which
knows what it loaded (see delegation._call_model_for)."""

from __future__ import annotations

import json
import sys
import urllib.request

_TIMEOUT_S = 300


def call_loaded_model(messages: list, tools: list) -> dict:
    module = sys.modules.get("routes.inference")
    backend = getattr(module, "_llama_cpp_backend", None) if module is not None else None
    if backend is None or not getattr(backend, "is_loaded", False):
        return {"content": "Error: no model is loaded to answer as the delegate.", "tool_calls": []}
    payload = {"model": backend.model_identifier or "local", "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    request = urllib.request.Request(
        f"{backend.base_url}/v1/chat/completions",
        data = json.dumps(payload).encode("utf-8"),
        headers = {"Content-Type": "application/json", **(backend._auth_headers or {})},
    )
    with urllib.request.urlopen(request, timeout = _TIMEOUT_S) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = (body.get("choices") or [{}])[0]
    return choice.get("message") or {"content": "", "tool_calls": []}
