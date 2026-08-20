# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-based image editing (img2img) using Studio's own diffusion backend.

Deliberately does NOT ship a second diffusion stack: Studio already loads and
manages diffusion models, so this tool inherits whatever the user has.

The real entry point is ``DiffusionBackend.generate()`` in
``core/inference/diffusion.py``. Passing ``init_image`` (a base64-encoded
source image) without a mask/upscale/reference selects the "img2img"
workflow inside that method, which denoises the source at ``strength``
toward ``prompt`` and returns the result at the SOURCE image's own size.
This is exactly what the production ``POST /images/generate`` route does
(``routes/inference.py``), so this tool goes through the same
``get_active_diffusion_engine()`` indirection rather than hard-coding the
diffusers backend directly.

That indirection is NOT full engine parity, though: it also covers the
native ``SdCppDiffusionBackend.generate()`` (``core/inference/sd_cpp_backend.py``),
which accepts the same keywords but rejects an image-conditioned call
outright -- passing ``init_image`` raises ``ValueError`` there ("img2img /
inpaint / reference / upscale are not yet supported on the native sd.cpp
engine"). So in practice: img2img only actually works on the diffusers
engine; on native sd.cpp this tool will raise that same ValueError. That is
the correct failure mode (it matches what the production route does), but
callers should not assume this tool edits regardless of which engine is
active.
"""

import base64
import io


def _studio_backend():
    """Return a callable(image_bytes=, prompt=, strength=) -> bytes.

    Imported lazily: diffusion pulls heavy libraries that must not load at
    module import time.
    """
    from core.inference.diffusion_engine_router import get_active_diffusion_engine

    def call(*, image_bytes, prompt, strength):
        engine = get_active_diffusion_engine()
        init_image = base64.b64encode(image_bytes).decode("ascii")
        result = engine.generate(prompt=prompt, init_image=init_image, strength=strength)
        buf = io.BytesIO()
        result["images"][0].save(buf, format="PNG")
        return buf.getvalue()

    return call


def edit_image(image_bytes, prompt, *, backend = None, strength = 0.6):
    """Apply a natural-language edit to ``image_bytes``. Returns PNG bytes."""
    if not prompt or not str(prompt).strip():
        raise ValueError("a prompt describing the edit is required")
    call = backend if backend is not None else _studio_backend()
    return call(image_bytes = image_bytes, prompt = str(prompt).strip(), strength = strength)
