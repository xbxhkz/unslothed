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

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

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

# sqlite-vec's vec0 native library -- found by LAUNCHING the exe (fix round 3)
# and reading stderr, not by any bundle-contents check:
#
#   RAG unavailable: sqlite-vec extension could not be loaded;
#   RAG features are disabled for this session (The specified module could not be found.)
#
# Note WHICH message that is. storage/rag_db.py:29-35 wraps `import sqlite_vec`
# in a try/except that logs "could not be imported"; the line above is the
# other one, from _warn_unavailable_once(), meaning the import SUCCEEDED and
# conn.load_extension() failed. That distinction is the whole diagnosis: the
# pure-Python sqlite_vec package is compiled into the PYZ (so it imports fine
# and appears nowhere on disk -- a file search for it returns nothing, which
# reads as "not bundled" and is misleading), while vec0.dll is package DATA
# that Analysis does not collect. Module present, payload absent: exactly the
# insightface shape from fix round 2.
#
# sqlite_vec.loadable_path() is path.join(path.dirname(__file__), "vec0"), and
# a PYZ module's synthetic __file__ is _MEIPASS/sqlite_vec/__init__.py, so the
# DLL must land in a "sqlite_vec" directory specifically -- SQLite appends the
# platform suffix (.dll) to the extensionless path itself.
#
# Failure mode without this: RAG (knowledge bases, hybrid retrieval,
# conversation archive) is silently off in the shipped installer while working
# in the dev tree, because the venv has the DLL. rag_db warns once and every
# caller degrades quietly by design -- so nothing surfaces it except reading
# startup stderr.
try:
    import sqlite_vec as _sqlite_vec_probe
    _vec0 = Path(_sqlite_vec_probe.__file__).parent / "vec0.dll"
    if _vec0.is_file():
        datas.append((str(_vec0), "sqlite_vec"))
    else:
        print(f"[Unslothed.spec] sqlite-vec present but vec0.dll missing at {_vec0}; "
              "RAG will be disabled in the build")
except Exception as _exc:  # noqa: BLE001 - optional dep, mirrors rag_db.py's own guard
    print(f"[Unslothed.spec] sqlite_vec not importable ({_exc}); RAG will be disabled in the build")

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

# hub.workers.* is spawned, never imported: hub/services/download_lifecycle.py
# runs it as [sys.executable, "-m", "hub.workers.hf_download", ...] and NOTHING
# in the tree imports it, so Analysis's import walk cannot see it and the whole
# subpackage was absent from the bundle -- verified against Analysis-00.toc,
# which listed 20+ other hub.* modules and no hub.workers at all.
#
# That made model downloads fail twice over: run.py's frozen `-m` shim answers
# the spawn, and this answers what the shim then has to import. Without both,
# a download dies with either an argparse usage dump or ModuleNotFoundError.
#
# collect_submodules rather than the one module, so a worker added upstream
# beside hf_download.py is bundled without anyone remembering this file.
hiddenimports += collect_submodules("hub.workers")

# torch and torchvision are EXCLUDED from Analysis below (see the excludes= arg)
# and copied in wholesale via datas instead. datas copies FILES; only Analysis
# builds the import GRAPH -- so their own submodules are present (copied) while
# anything EXTERNAL they import was never walked and is simply absent.
#
# torchvision/__init__.py line 3 does `from modulefinder import Module`. That is
# pure-Python stdlib, so it lives in the PYZ or nowhere, and nowhere is what it
# was. Loading a Wan pipeline died three levels deep:
#   transformers/image_utils.py -> torchvision -> ModuleNotFoundError: modulefinder
# resurfacing at the top as "Could not import module 'UMT5EncoderModel'", which
# names nothing involved. warn-Unslothed.txt was silent because Analysis never
# looked at torchvision at all.
#
# Swept torch + torchvision (2009 files) for external imports absent from the
# TOC: 23 candidates, of which 19 are interpreter builtins (always present), two
# are optional and uninstalled (annotationlib, defusedxml), win32api is bundled
# already and guarded by try/except anyway -- leaving exactly this one.
hiddenimports += ["modulefinder"]

# Distribution METADATA for the packages diffusers and transformers version-check
# at import time. PyInstaller bundles modules by following imports, but a
# package's .dist-info is a separate artifact it copies only when some hook asks
# -- so `import requests` worked while importlib.metadata.version("requests")
# raised, and selecting a diffusion model died with
#   The 'requests' distribution was not found and is required by this application
# (diffusers/dependency_versions_check.py, whose hint text still says
# "pip install transformers -U" because diffusers copied the checker verbatim).
#
# Both libraries run these checks at IMPORT time, so a gap is not reachable
# until the moment a model is selected -- long after any startup smoke test.
#   diffusers:    python requests filelock numpy
#   transformers: tqdm regex packaging filelock numpy tokenizers
#                 huggingface-hub safetensors accelerate pyyaml
#
# Beyond those declared lists, any unguarded importlib.metadata read that runs at
# import time is the same bug. transformers/audio_utils.py:55 is the worked
# example, and it shows why the two notions of "installed" diverge when frozen:
#
#     if is_torchcodec_available():      # find_spec() -> the MODULE is bundled: True
#         version.parse(importlib.metadata.version("torchcodec"))   # METADATA: raises
#
# It surfaced as "Could not import module 'UMT5EncoderModel'" while loading a Wan
# pipeline, because transformers' lazy-module __getattr__ swallows the real
# exception and re-raises a generic one -- the message names neither torchcodec
# nor metadata.
#
# The list below is therefore NOT hand-picked. It is every distribution that a
# package in the model-loading chain (transformers, diffusers, accelerate, peft,
# datasets, timm, unsloth_zoo, ...) passes to importlib.metadata.version /
# .distribution / .metadata or pkg_resources.get_distribution, AND which is
# actually installed -- swept mechanically, not guessed. Of those, only
# `torchcodec` and `gguf` were missing; the other 25 were present only
# INCIDENTALLY, copied by some other package's hook. Listing them all makes that
# a guarantee rather than a coincidence a dependency bump could withdraw.
#
# copy_metadata raises on an absent package, so a build fails loudly here rather
# than shipping an exe that dies on model load. Metadata is a few KB apiece, so
# being over-inclusive is close to free; a miss costs a full rebuild.
_METADATA_CHECKED_AT_IMPORT = [
    # diffusers + transformers dependency_versions_check (declared lists)
    "requests",
    "filelock",
    "numpy",
    "tqdm",
    "regex",
    "packaging",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "pyyaml",
    # swept: read via importlib.metadata at import time somewhere in the chain
    "torchcodec",       # transformers/audio_utils.py:55 -- was missing
    "gguf",             # diffusers -- was missing
    "bitsandbytes",
    "datasets",
    "dill",
    "duckdb",
    "fsspec",
    "kernels",
    "opentelemetry-api",
    "pandas",
    "peft",
    "polars",
    "pyarrow",
    "torch",
    "torchao",
    "transformers",
]
for _dist in _METADATA_CHECKED_AT_IMPORT:
    try:
        datas += copy_metadata(_dist)
    except Exception as exc:  # noqa: BLE001 -- surface the dist name, then stop
        raise SystemExit(
            f"[Unslothed.spec] copy_metadata({_dist!r}) failed: {exc}. "
            "diffusers/transformers version-check this at import, so a frozen "
            "build without its .dist-info dies when a model is selected. Install "
            "it into the build venv rather than removing it from this list."
        ) from exc

# Packages that open a data file NEXT TO their own source at import time.
# PyInstaller follows imports, so the module lands in the PYZ while the data file
# beside it is simply left behind -- the module imports fine and then raises on
# the open, which reads as a broken feature rather than a packaging fault.
#
# kernels: deps.py does
#     with open(Path(__file__).parent / "python_depends.json") as f
# at import. Loading a Wan/diffusion pipeline died with
#     Failed to import diffusers.pipelines.wan.pipeline_wan ...
#     Cannot load dependency data, is `kernels` correctly installed?
# which names the wrong culprit: kernels IS installed, its json was not shipped.
#
# whisper: normalizers/english.py opens english.json, and assets/ holds
# mel_filters.npz plus the two .tiktoken vocabularies -- whisper cannot compute
# a mel spectrogram or tokenize without them. Imported by
# core/training/trainer.py:2286, so this was a latent second instance of the
# same bug; found by sweeping site-packages for the pattern rather than by
# waiting for it to crash. ~1.7 MB.
#
# These land inside _internal/, which is where Path(__file__).parent resolves
# for a PYZ module -- unlike the frontend, which run.py resolves ABOVE _internal/
# and which therefore needs the post-build copy in build-installer.ps1 instead.
datas += collect_data_files("kernels")
datas += collect_data_files("whisper")

# Triton, complete. Video generation died mid-run with
#   RuntimeError: 0 active drivers ([]). There should only be one.
# from triton/runtime/driver.py, reached because diffusers' GGUF dequant path
# (quantizers/gguf/utils.py forward_native) goes through torch.compile ->
# inductor -> triton codegen.
#
# Three separate omissions, each of which alone still fails -- established by
# hot-patching a built _internal/ and re-probing after each one:
#
#  1. ENTRY POINTS. triton/backends/__init__.py discovers backends via
#     entry_points().select(group="triton.backends"), which lives in the
#     dist-info. No metadata -> zero backends -> "0 active drivers". The dist is
#     named triton-windows, not triton. (There is a TRITON_BACKENDS_IN_TREE=1
#     filesystem-scan fallback; not used, because depending on a fast-path env
#     var is more fragile than shipping the metadata its default path wants.)
#  2. BACKEND FILES. Analysis pulled in only __init__.py and driver.py. A
#     backend needs compiler.py too, plus driver.c (triton compiles a driver
#     shim from that source at RUNTIME) and bin/ptxas.exe, include/, lib/.
#  3. LAZY SUBMODULES. With the above fixed, codegen still failed on
#     `import triton.language.extra.cuda` inside nvidia/compiler.py -- imported
#     lazily, so the static walk never saw it.
#
# Verified end to end inside the built exe, not merely as bundled files: a real
# torch.compile(backend="inductor") round-trip codegens, launches on the GPU and
# matches eager exactly (|delta| = 0.0) on an RTX 4050 (sm_89). ~129 MB.
#
# NOTE the app's own guard (core/export/worker.py:576) is `import triton`
# succeeding. Frozen, that passed while triton was unusable -- the same
# false-positive shape as is_torchcodec_available(). It is honest again only
# because triton genuinely works now; it would not have caught this.
hiddenimports += collect_submodules("triton")
datas += collect_data_files("triton")
datas += copy_metadata("triton-windows")

# diceware generates the bootstrap admin password on a FIRST RUN with no
# password set (auth/storage.py's generate_bootstrap_password). Found by
# launching the built exe against an EMPTY UNSLOTH_STUDIO_HOME, which is the
# only configuration that reaches this code -- every earlier test pointed at a
# home that already had a password, so the crash never fired:
#
#   ModuleNotFoundError: No module named 'diceware.random_sources'
#     ... diceware/__init__.py:84 in get_random_sources
#
# diceware resolves its random sources through an entry-point-style spec dict
# in __about__.py and imports them by name, so Analysis's static walk never
# follows the edge. collect_submodules picks up random_sources (and anything
# else it loads the same way); collect_data_files brings the wordlists, which
# are equally invisible and equally required -- a random source with no word
# list still cannot produce a passphrase.
#
# Severity note, because it is easy to under-read: without this a FRESH INSTALL
# crashes on first launch. An upgrade over an existing home does not, which is
# exactly why it survived nine earlier verification passes.
hiddenimports += collect_submodules("diceware")
datas += collect_data_files("diceware")

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

# The sibling scripts named above are DELIBERATELY NOT bundled, and this note
# exists so nobody "fixes" that by adding datas entries for them. It would
# achieve nothing.
#
# utils/prebuilt/update_flow.py:135 runs them as
#     cmd = [sys.executable, str(script), "--resolve-prebuilt", "latest", ...]
# In a frozen build sys.executable is Unslothed.exe, not a Python interpreter,
# and there is no python.exe in a PyInstaller onedir layout to substitute. So
# the invocation cannot work however the script is packaged.
#
# Measured, not assumed -- running the built exe the way sys.executable would:
#     Unslothed.exe some_installer.py --resolve-prebuilt latest --output-format json
#     -> exit code 2
#     Unslothed.exe: error: unrecognized arguments: some_installer.py ...
# run.py's argparse rejects the extra arguments, so the subprocess exits
# non-zero, resolve_prebuilt_for_host's except-clause fires and returns None --
# the same result as the script not being found at all. Bundling would move the
# failure from "no script" to "argparse error" and change nothing a user sees.
#
# What this costs: in the installed build, the in-app "check for a newer
# llama.cpp / whisper.cpp prebuilt" flow always reports nothing available. It
# fails OPEN by design (utils/prebuilt/update_flow.py:145 -- "any error -> None
# so a source build never blocks the app"), so nothing crashes and the bundled
# binaries keep working; only self-update is inert. Closing it properly needs an
# interpreter the frozen app can invoke, or an in-process import path replacing
# the subprocess -- both changes to upstream files this fork does not edit.

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
