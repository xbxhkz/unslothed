# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-based image editing, driven by Studio's own diffusion backend.

Unlike the other vision tools this is not a port -- Assist drove its own
sd-server, and this drives Studio's existing img2img. The backend is
injectable so these tests pin the contract (the prompt and the source image
both reach it, and its result is what comes back) without loading a
diffusion model.
"""

import io

import pytest
from PIL import Image

from core.inference.assist_vision.image_edit import edit_image


def _png(size=(32, 32), color=(10, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Backend:
    def __init__(self):
        self.calls = []

    def __call__(self, *, image_bytes, prompt, strength):
        self.calls.append({"prompt": prompt, "strength": strength, "size": len(image_bytes)})
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (240, 30, 30)).save(buf, format="PNG")
        return buf.getvalue()


class TestEditImage:
    def test_the_prompt_and_source_image_reach_the_backend(self):
        backend = _Backend()
        edit_image(_png(), "make the sky orange", backend=backend)
        assert backend.calls[0]["prompt"] == "make the sky orange"
        assert backend.calls[0]["size"] > 0

    def test_the_backend_result_is_returned_not_the_original(self):
        """Fails if the source image is passed through untouched."""
        out = edit_image(_png(), "anything", backend=_Backend())
        assert Image.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0)) == (240, 30, 30)

    def test_an_empty_prompt_is_rejected(self):
        with pytest.raises(ValueError):
            edit_image(_png(), "   ", backend=_Backend())

    def test_strength_is_forwarded(self):
        backend = _Backend()
        edit_image(_png(), "x", backend=backend, strength=0.25)
        assert backend.calls[0]["strength"] == 0.25

    def test_the_adapter_uses_keywords_the_real_backend_accepts(self):
        """Guards against drift: the suite's double cannot catch a typo'd or
        renamed keyword in the real call, so assert the real signature instead."""
        import inspect

        from core.inference.diffusion import DiffusionBackend

        params = set(inspect.signature(DiffusionBackend.generate).parameters)
        assert {"prompt", "init_image", "strength"} <= params


class TestRealBackendConformance:
    """The half of the contract the signature check missed.

    Introspecting ``DiffusionBackend.generate`` says nothing about
    ``get_active_diffusion_engine`` -- the symbol ``_studio_backend()`` actually
    calls -- nor about the ``result["images"][0]`` shape the adapter indexes
    into. Rename either upstream and the whole suite stayed green while the tool
    was dead. These assert the real symbols and the real return shape.
    """

    def test_the_engine_accessor_this_tool_calls_exists_and_is_callable(self):
        from core.inference import diffusion_engine_router

        fn = getattr(diffusion_engine_router, "get_active_diffusion_engine", None)
        assert fn is not None, (
            "image_edit._studio_backend() imports this name; renaming it "
            "upstream breaks the tool"
        )
        assert callable(fn)

    def test_the_accessor_takes_no_required_arguments(self):
        """The adapter calls it bare."""
        import inspect

        from core.inference.diffusion_engine_router import get_active_diffusion_engine

        required = [
            p for p in inspect.signature(get_active_diffusion_engine).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        assert not required, f"accessor now requires {[p.name for p in required]}"

    def test_the_real_backend_returns_an_images_list_the_adapter_can_index(self):
        """Pins the shape ``result["images"][0]`` depends on, read off the real
        generate() rather than off the test double."""
        import ast
        import inspect
        import textwrap

        from core.inference.diffusion import DiffusionBackend

        src = textwrap.dedent(inspect.getsource(DiffusionBackend.generate))
        tree = ast.parse(src)
        dict_returns = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        ]
        assert dict_returns, "generate() no longer returns a dict literal"

        for returned in dict_returns:
            keys = [
                k.value for k in returned.keys
                if isinstance(k, ast.Constant)
            ]
            assert "images" in keys, (
                'generate() returns a dict without an "images" key; '
                'image_edit indexes result["images"][0]'
            )
            value = returned.values[keys.index("images")]
            # list(...) -- indexable by [0], which is what the adapter does.
            assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == "list", (
                '"images" is no longer built as a list; result["images"][0] '
                "may no longer be valid"
            )

    def test_the_adapter_saves_what_the_backend_puts_in_images(self):
        """Exercises _studio_backend()'s real body against a stand-in engine,
        so the indexing and the PNG save are executed rather than assumed."""
        from core.inference import diffusion_engine_router
        from core.inference.assist_vision.image_edit import _studio_backend

        class _Engine:
            def __init__(self):
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return {"images": [Image.new("RGB", (8, 8), (7, 8, 9))], "seed": 1}

        engine = _Engine()
        original = diffusion_engine_router.get_active_diffusion_engine
        diffusion_engine_router.get_active_diffusion_engine = lambda: engine
        try:
            out = _studio_backend()(image_bytes = _png(), prompt = "x", strength = 0.5)
        finally:
            diffusion_engine_router.get_active_diffusion_engine = original

        assert Image.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0)) == (7, 8, 9)
        # The source image reaches the engine base64-encoded, selecting img2img.
        assert engine.kwargs["init_image"]
        assert engine.kwargs["strength"] == 0.5
