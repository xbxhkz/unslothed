# packaging/Unslothed.spec
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# PyInstaller spec for the self-contained Unslothed Windows build.
#
# Run via packaging/build-installer.ps1, not directly -- the frontend must be
# built first or the app ships with no UI.

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# SPECPATH is the directory holding this file: <repo>/packaging. One .parent
# reaches the repo root. A second .parent (as an earlier draft of this file
# had) overshoots to the folder *containing* the repo, which quietly turns
# every path below into a wrong one -- the build then fails with "not found"
# errors that look unrelated to this line. Print it so a future slip is
# visible immediately instead of as a confusing downstream failure.
ROOT = Path(SPECPATH).resolve().parent
BACKEND = ROOT / "studio" / "backend"
print(f"[Unslothed.spec] ROOT    = {ROOT}")
print(f"[Unslothed.spec] BACKEND = {BACKEND}")

# The real, runnable entry point. studio/backend/main.py only defines the
# FastAPI `app` object (no __main__ guard, no server start) -- it is meant to
# be imported, e.g. by `uvicorn main:app`, not executed. studio/backend/run.py
# is the self-contained script: it has the `if __name__ == "__main__":` block,
# does startup-order-sensitive setup (CPU thread caps, ROCm stubs, std-stream
# normalization before structlog binds them) ahead of any heavy import, and
# only then does `from main import app` inside run_server(). Freezing main.py
# instead would produce an exe that imports and exits without ever serving.
ENTRY = BACKEND / "run.py"
assert ENTRY.is_file(), (
    f"Entry point not found at {ENTRY}. Expected studio/backend/run.py -- "
    "if the backend was restructured, point this at whatever file now has "
    "the `if __name__ == \"__main__\":` block that calls run_server()."
)
print(f"[Unslothed.spec] entry   = {ENTRY}")

# core.inference.assist_vision / assist_code are imported at runtime as bare
# top-level names (`from core.inference.assist_vision import ...` in
# core/inference/tools.py), not as `studio.backend.core.inference...` --
# run.py puts BACKEND on sys.path itself (`sys.path.insert(0, backend_dir)`),
# never ROOT. The two dotted paths both happen to be importable (studio/ and
# studio/backend/ both have __init__.py), but they resolve to two different
# module objects; collect_submodules() on the studio.backend-prefixed name
# would bundle a parallel identity nothing ever imports, leaving the real
# bare-named submodules unprotected. Put BACKEND on sys.path here too so
# collect_submodules can resolve the same bare names the app actually uses.
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# --- data ------------------------------------------------------------------
# The frontend dist is the UI. Without it the app starts and serves nothing,
# which looks like a backend fault and is not. PyInstaller itself fails loudly
# at build time if any of these three source paths do not exist.
datas = [
    (str(ROOT / "studio" / "frontend" / "dist"), "studio/frontend/dist"),
    (str(BACKEND / "assets"), "studio/backend/assets"),
    (str(BACKEND / "requirements"), "studio/backend/requirements"),
]
for optional in ("vendor", "plugins"):
    src = BACKEND / optional
    if src.is_dir():
        datas.append((str(src), f"studio/backend/{optional}"))
    else:
        print(f"[Unslothed.spec] optional data dir {optional} not present at {src}, skipping")

# --- frozen __file__/parents mismatches -------------------------------------
# Four instances of one bug class surfaced one at a time across this task:
# torch, insightface, sd_cpp, and now this one (routes/preview.py crashing the
# app at startup). The shape is always the same: a module does
# Path(__file__).parent[...] / "some/relative/resource", which is correct in
# the dev tree (nesting matches studio/backend/<module path>) but wrong once
# frozen, because PyInstaller gives frozen modules a synthetic __file__ rooted
# at sys._MEIPASS using the module's DOTTED IMPORT NAME (e.g. "routes.preview"
# -> _internal/routes/preview.py), not its original on-disk nesting under
# studio/backend/. A chain of N parents that correctly reaches studio/backend/
# in the dev tree reaches somewhere else entirely once frozen -- sometimes
# still inside _internal/ (fixable with an extra flat datas entry, same
# pattern as insightface), sometimes one level *above* _internal/, outside
# anywhere a datas entry can reach at all (that shape needs the sd_cpp-style
# runtime-hook treatment, or a post-build copy -- not attempted here).
#
# Swept every __file__-relative resource resolution under studio/backend/
# (excluding tests/, never shipped) for this task's fix round 3. Findings and
# what's fixed here vs. reported-only are in packaging's task-4-report.md;
# summary of what's fixed below.

# routes/preview.py's _PREVIEW_PAGE_HTML does
# (Path(__file__).resolve().parent.parent / "assets" / "preview_page.html").read_text(...)
# at MODULE LEVEL -- this is the confirmed startup crash. Frozen, parent.parent
# from _internal/routes/preview.py lands on _internal/ (flat), not
# _internal/studio/backend/ (nested, where the datas entry above places
# studio/backend/assets/). Same flat "assets" destination also fixes three
# further, non-crashing but real functional gaps found in the sweep, all
# reading paths built the same way, all lazy (inside a function, not at
# import time, so none of them crash startup on their own -- confirmed by
# reading each call site before relying on that):
#   - core/inference/chat_templates.py's _ASSETS_DIR (assets/chat_templates/)
#   - utils/inference/inference_config.py's inference_defaults.json and
#     assets/configs/model_defaults/ lookups
#   - utils/models/model_config.py's assets/configs/model_defaults/ lookup
# Keeping the nested "studio/backend/assets" entry above too, on the
# insightface precedent: a second consumer using the dev-mode-style nested
# path (Path(__file__).parent-relative, not sys._MEIPASS-relative) could
# exist without having been traced by this sweep.
datas.append((str(BACKEND / "assets"), "assets"))

# utils/native_tls.py's _VENDOR_DIR (str(Path(__file__).resolve().parent.parent
# / "vendor")) is the same flat-vs-nested mismatch, for the vendored
# `truststore` package activate_native_tls() puts on sys.path. Only add the
# flat copy if vendor/ exists at all (mirrors the optional-loop guard above).
if (BACKEND / "vendor").is_dir():
    datas.append((str(BACKEND / "vendor"), "vendor"))

# core/inference/tools.py's _SANDBOX_SITE_DIR (placed on a sandboxed child
# process's PYTHONPATH for the code-interpreter tool) and
# core/data_recipe/local_callable_validators.py's _OXC_TOOL_DIR (the OXC
# JS/TS validator, invoked via `node validate.mjs`) both use a SINGLE
# Path(__file__).parent -- their own containing directory, not a multi-parent
# chain -- so dev and frozen resolution agree with each other (both land on
# "wherever core/inference/ or core/data_recipe/ ends up"). The bug for these
# two is different: neither directory is bundled via ANY datas entry at all
# yet (core/inference/*.py and core/data_recipe/*.py are pure-Python and get
# compiled into the PYZ rather than kept as loose files, but sandbox_site/ and
# oxc-validator/ hold non-.py payloads -- a sitecustomize.py shim a *subprocess*
# needs to find as a real file, and a .mjs script Node.js must read directly --
# that Analysis never collects on its own). Both are small (tens of KB); add
# them at the nested destination matching their single-parent frozen
# resolution (_internal/core/inference/sandbox_site,
# _internal/core/data_recipe/oxc-validator).
_sandbox_site = BACKEND / "core" / "inference" / "sandbox_site"
if _sandbox_site.is_dir():
    datas.append((str(_sandbox_site), "core/inference/sandbox_site"))
else:
    print(f"[Unslothed.spec] sandbox_site not found at {_sandbox_site}, skipping")

_oxc_validator = BACKEND / "core" / "data_recipe" / "oxc-validator"
if _oxc_validator.is_dir():
    datas.append((str(_oxc_validator), "core/data_recipe/oxc-validator"))
else:
    print(f"[Unslothed.spec] oxc-validator not found at {_oxc_validator}, skipping")

# torch and its CUDA payload are copied wholesale rather than analysed. Torch
# loads extensions dynamically and ships CUDA DLLs that static analysis never
# sees; letting PyInstaller try produces a build that imports and then fails
# at the first real call.
#
# The install location is found from the package itself, not from
# site.getsitepackages() -- in a venv that list's last entry is not reliably
# the venv's own site-packages, and the whole torch strategy depends on
# finding the real directory. A wrong path here means src.is_dir() below is
# silently False and torch is silently left out of the bundle: it still
# builds, it still starts, and it dies with ModuleNotFoundError on the first
# real tool call -- the most expensive place to discover a packaging bug.
_torch_spec = importlib.util.find_spec("torch")
assert _torch_spec is not None and _torch_spec.origin, (
    "torch is not importable in the environment running this spec "
    f"(sys.executable={sys.executable}). Build with the same Python "
    "environment the app runs against -- one with torch installed -- not a "
    "bare system interpreter."
)
SITE = Path(_torch_spec.origin).parent.parent
assert SITE.is_dir(), (
    f"Derived site-packages={SITE} from torch's spec.origin="
    f"{_torch_spec.origin!r}, but that directory does not exist."
)
print(f"[Unslothed.spec] SITE (site-packages, via torch's spec.origin) = {SITE}")

# Windows CUDA wheels (the "+cuXXX" local version) bundle their CUDA payload
# INSIDE torch/lib/ -- there is no separate nvidia-* package to check the way
# there would be on Linux, so nvidia's absence from the heavy-tree loop below
# is expected and not itself a signal of anything wrong. What *is* checked
# here: whether this interpreter's torch is a CUDA build at all. A CPU-only
# interpreter (e.g. system Python instead of the Studio venv) passes every
# assert above and produces a complete, working-looking bundle that silently
# never uses the GPU -- nothing downstream catches this, since the smoke test
# only asserts that tools register, not which torch is underneath.
_torch_version = importlib.metadata.version("torch")
assert "+cu" in _torch_version, (
    f"torch {_torch_version} in this environment (sys.executable="
    f"{sys.executable}) is not a CUDA build (no '+cuXXX' tag). This spec "
    "bundles torch wholesale and ships it as the app's only torch -- "
    "building from a CPU-only interpreter would silently produce a "
    "complete, working-looking app that never uses the GPU, and nothing "
    "downstream would catch it (the smoke test only checks that tools "
    "register). Activate the Studio venv (torch 2.10.0+cu130), not the "
    "system interpreter."
)
print(f"[Unslothed.spec] torch version = {_torch_version} (CUDA build confirmed)")

# torch and torchvision are load-bearing -- hiddenimports below assumes both
# are on disk (assist_vision calls torchvision.models.detection and
# torchvision.transforms.functional). Missing either must fail the build, not
# ship silently broken. nvidia/functorch/torchgen are copied when present but
# are not asserted: nvidia is absent on a legitimate CPU-only torch install
# (this dev machine has one), and functorch/torchgen are torch-internal
# support trees whose layout has moved between torch versions.
_REQUIRED_HEAVY = {"torch", "torchvision"}
for heavy in ("torch", "torchvision", "nvidia", "functorch", "torchgen"):
    src = SITE / heavy
    if src.is_dir():
        print(f"[Unslothed.spec] copying {heavy} wholesale from {src}")
        datas.append((str(src), heavy))
    elif heavy in _REQUIRED_HEAVY:
        raise AssertionError(
            f"Required tree '{heavy}' not found at {src}. The frozen build "
            "would ship without it and fail at the first call that needs it, "
            "not at build time. Install it in this environment before "
            "building."
        )
    else:
        print(f"[Unslothed.spec] {heavy} not found at {src}, skipping (optional)")

# --- hidden imports --------------------------------------------------------
# Every entry below is imported INSIDE a function body somewhere in
# assist_vision or assist_code, deliberately, to keep startup fast. That is
# exactly what PyInstaller's static analysis cannot see. Without these the
# frozen build registers the tool and then raises ModuleNotFoundError on the
# first call -- a failure that looks like a tool bug, not a packaging bug.
hiddenimports = [
    # assist_vision
    "cv2",
    "numpy",
    "onnxruntime",
    "torch",
    "torchvision",
    "torchvision.models.detection",
    "torchvision.transforms.functional",
    "ultralytics",
    "insightface",
    "insightface.app",
    "insightface.model_zoo",
    "PIL",  # `from PIL import Image` inside webcam.py and paths.py
    # assist_code needs no third-party imports -- it is stdlib only, and its
    # language servers are external processes installed on first use.
]
hiddenimports += collect_submodules("core.inference.assist_vision")
hiddenimports += collect_submodules("core.inference.assist_code")

# insightface ships its own non-.py data under its package tree (notably
# data/objects/meanshape_68.pkl, read by model_zoo/landmark.py's Landmark
# class via insightface/data/pickle_object.py's get_object() -- a helper that
# is itself PyInstaller-aware, branching on `sys.frozen`/`sys._MEIPASS`, which
# is a strong signal upstream expects exactly this kind of freeze). cv2,
# onnxruntime and ultralytics all get their data files collected automatically
# because pyinstaller-hooks-contrib ships hooks for them; it ships none for
# insightface, so without this explicit collection those files are silently
# absent from the bundle. face_swap.py's _get_analyzer() calls
# FaceAnalysis(name=_MODEL_PACK_NAME, ...) with no allowed_modules
# restriction, which loads the landmark_3d_68 submodel and therefore this
# exact file on first real use -- not a hypothetical, a reachable runtime path.
datas += collect_data_files("insightface")

# The collect_data_files() call above is NOT sufficient on its own -- found
# and confirmed with a standalone PyInstaller 6.20.0 onedir probe build
# (mirroring get_object()'s exact logic) after the first attempt at this fix
# turned out to still be broken. insightface/data/pickle_object.py's
# get_object() branches on sys.frozen: when true it reads from
# <sys._MEIPASS>/objects/<name>.pkl -- FLAT, directly under the frozen app's
# root -- not the package-nested <sys._MEIPASS>/insightface/data/objects/
# path collect_data_files() produces (that nested layout only serves
# get_object()'s *unfrozen* branch, Path(__file__).parent). Without this
# second, flat copy, the file collect_data_files() placed is real but at a
# path get_object() never looks at when frozen, so Landmark still gets
# mean_lmk=None. Kept collect_data_files() above too, in case some other
# insightface internal uses the unfrozen-style nested lookup even when frozen.
_insightface_spec = importlib.util.find_spec("insightface")
if _insightface_spec and _insightface_spec.submodule_search_locations:
    _insightface_root = Path(list(_insightface_spec.submodule_search_locations)[0])
    _insightface_objects = _insightface_root / "data" / "objects"
    if _insightface_objects.is_dir():
        datas.append((str(_insightface_objects), "objects"))
        print(f"[Unslothed.spec] copying insightface objects flat from {_insightface_objects}")
    else:
        print(f"[Unslothed.spec] insightface data/objects not found at {_insightface_objects}, skipping flat copy")
else:
    print("[Unslothed.spec] insightface package spec not found, skipping flat objects copy")

# core/inference/sd_cpp_backend.py's _installer_module() does
# sys.path.insert(0, studio_dir) at runtime and then `import
# install_sd_cpp_prebuilt` -- a local script at studio/install_sd_cpp_prebuilt.py,
# not a pip package. The path is added at runtime, so Analysis's static walk
# never sees this import and nothing above covers it: same unbundled-payload
# shape as insightface's data files, one level up (a whole file instead of a
# package's data). Every call site catches the resulting ModuleNotFoundError
# and fails open (ensure_sd_cpp_binary returns None, falls back to diffusers),
# so this does not crash -- but on a CPU-only install, or with
# UNSLOTH_DIFFUSION_ENGINE=sd_cpp, GGUF-format diffusion models silently and
# permanently lose the native engine. Sibling scripts under studio/ (audited:
# install_llama_prebuilt.py, install_whisper_prebuilt.py, install_manifest.py,
# install_python_stack.py, install_node_prebuilt.py) are invoked via
# subprocess + a filesystem search, not this sys.path.insert+import pattern,
# so they are not the same bug -- only this one is.
#
# This datas entry alone does NOT fix it -- confirmed with the same
# standalone PyInstaller 6.20.0 onedir probe that caught the insightface gap
# above. _installer_module() computes studio_dir as
# Path(__file__).resolve().parents[3], which in the dev tree lands on
# studio/ (four levels up from studio/backend/core/inference/<file>.py). In a
# PyInstaller 6.x onedir frozen build it does not: frozen modules get a
# synthetic __file__ rooted at sys._MEIPASS (the _internal/ folder), so
# parents[3] from _internal/core/inference/<file>.py lands one level ABOVE
# _internal -- the top-level onedir folder next to Unslothed.exe -- while
# this datas entry (like every datas entry) lands inside _internal/studio/,
# per PyInstaller's COLLECT convention that nothing but the exe itself can be
# placed outside _internal/. The two paths never meet, so a datas entry by
# itself cannot close this gap; see hook_runtime_sd_cpp_installer.py below,
# which does, by putting the correct directory on sys.path before
# _installer_module() runs its own (here, harmless-but-wrong) insert.
datas.append((str(ROOT / "studio" / "install_sd_cpp_prebuilt.py"), "studio"))

a = Analysis(
    [str(ENTRY)],
    pathex = [str(ROOT), str(BACKEND)],
    binaries = [],
    datas = datas,
    hiddenimports = hiddenimports,
    hookspath = [],
    # See hook_runtime_sd_cpp_installer.py's own header comment: this is what
    # actually closes the sd_cpp installer-script gap described above the
    # datas.append() near the top of this file -- the datas entry alone is
    # not sufficient in a PyInstaller 6.x onedir build.
    runtime_hooks = [str(Path(SPECPATH).resolve() / "hook_runtime_sd_cpp_installer.py")],
    # Excluded because they are copied in via `datas` above. Leaving them in
    # Analysis doubles the build time and produces a broken CUDA payload.
    excludes = ["torch", "torchvision", "nvidia", "functorch", "torchgen"],
    noarchive = False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries = True,
    name = "Unslothed",
    debug = False,
    strip = False,
    upx = False,          # UPX corrupts CUDA DLLs; never enable it here.
    console = True,       # the backend logs to stdout; a windowed build hides startup failures
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip = False,
    upx = False,
    name = "Unslothed",
)
