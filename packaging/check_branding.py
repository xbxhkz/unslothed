# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Assert the branding changed where it should and nowhere else.

Run from the repo root:  python packaging/check_branding.py

The risk this guards is not "did we rename enough" but "did we rename too
much". Of 3089 `unsloth` occurrences in the frontend, 2351 are import paths,
API routes and identifiers. Renaming those breaks the app, and the breakage
would not be obvious from a diff.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "studio" / "frontend"
LOCALES = FRONTEND / "src" / "i18n" / "locales"

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


# 1. The page title.
index_html = (FRONTEND / "index.html").read_text(encoding = "utf-8")
check("<title>Unslothed</title>" in index_html,
      "index.html: <title> is not Unslothed")

# 2. Tauri product name and window title -- but NOT the identifier.
conf = json.loads((ROOT / "studio" / "src-tauri" / "tauri.conf.json").read_text(encoding = "utf-8"))
check(conf.get("productName") == "Unslothed",
      f"tauri.conf.json: productName is {conf.get('productName')!r}, expected 'Unslothed'")
titles = [w.get("title") for w in (conf.get("app") or {}).get("windows") or []]
check(all(t == "Unslothed" for t in titles),
      f"tauri.conf.json: window titles are {titles!r}, expected all 'Unslothed'")

# The identifier is the OS-level application identity. Changing it makes
# Windows and macOS treat an upgrade as a different app, orphaning settings.
check(conf.get("identifier") == "ai.unsloth.studio",
      f"tauri.conf.json: identifier is {conf.get('identifier')!r} -- it MUST stay "
      "'ai.unsloth.studio'; changing it orphans user settings on upgrade")

# 3. Nothing that is not a user-visible string may have changed. If any import
#    path, API route or CSS class picked up the new name, the app is broken.
DANGEROUS = [
    (re.compile(r"from\s+['\"][^'\"]*unslothed[^'\"]*['\"]"), "an import path"),
    (re.compile(r"['\"]/api/[^'\"]*unslothed[^'\"]*['\"]"), "an API route"),
    (re.compile(r"ai\.unslothed\."), "the Tauri identifier"),
]
for path in FRONTEND.joinpath("src").rglob("*.ts*"):
    text = path.read_text(encoding = "utf-8", errors = "replace")
    for pattern, what in DANGEROUS:
        if pattern.search(text):
            failures.append(f"{path.relative_to(ROOT)}: {what} was renamed -- this breaks the app")

# 4. The AGPL attribution names upstream and must survive.
en = (LOCALES / "en.ts").read_text(encoding = "utf-8")
check("Unsloth AI Inc." in en,
      "en.ts: the AGPL attribution to 'Unsloth AI Inc.' was removed -- that is "
      "upstream's legal notice, not product branding")

# 5. The brand is untranslated: every locale carries the same literal.
missing = [p.name for p in sorted(LOCALES.glob("*.ts"))
           if "Unslothed" not in p.read_text(encoding = "utf-8", errors = "replace")]
check(not missing, f"locales missing the 'Unslothed' literal: {missing}")

# 6. Under-renaming: catch a leftover "Unsloth" sitting in a user-visible string
#    or JSX text node, outside the 12 locale files (those are covered by #4/#5).
#    Checks #1-#5 above only look at four already-known surfaces; they would
#    stay green even if hundreds of hardcoded brand strings were never touched
#    -- which is exactly what happened the first time this script shipped: it
#    passed while ~80 such strings sat unrenamed in studio/frontend/src.
#
#    Every surviving standalone "Unsloth" (not "Unslothed", not embedded in a
#    camelCase identifier like `classifyUnslothSupport`) must be on this
#    allowlist, which is the KEEP tables from
#    .superpowers/sdd/2026-08-28-unslothed-packaging/brand-classification.md,
#    reduced to one matchable substring each. Every entry carries the reason
#    it is permitted -- that reason is the part future readers actually need;
#    a hit that is not on this list is a real, unfixed leftover, not noise.
#
#    A camelCase identifier such as `classifyUnslothSupport` never satisfies
#    \bUnsloth\b in the first place (there is no boundary between "y" and "U",
#    or between "h" and "S"), so those ~40 code-only hits need no allowlist
#    entry at all; only standalone-word occurrences reach this list.
BRAND_WORD = re.compile(r"\bUnsloth\b(?!ed)")

# (file suffix relative to src/, or None for any file, substring to match, reason)
KEEP_ALLOWLIST = [
    ("components/tauri/startup-messages.ts",
     "loading PyTorch, Unsloth and Transformers",
     "matches backend/run.py's printed startup literal verbatim; renaming "
     "silently breaks the startup-message state machine"),
    ("features/chat/mcp-composer-button.tsx",
     "Unsloth Docs",
     "label for an MCP preset pointing at the real external site "
     "https://unsloth.ai/docs -- upstream's site, not this app"),
    ("features/chat/prompt-storage/prompt-storage-dialog.tsx",
     "ShareGPT format for Unsloth fine-tuning",
     "names the upstream fine-tuning technique/library, not this product"),
    ("features/chat/tour/steps.tsx",
     "Recommended is Unsloth",
     "describes the Hub \"Unsloth\" tab, an HF-publisher filter"),
    ("features/chat/tour/steps.tsx",
     "Search Unsloth",
     "describes the Hub \"Unsloth\" tab, an HF-publisher filter"),
    ("features/export/export-page.tsx",
     "upstream Unsloth imatrix",
     "text says \"upstream\" explicitly"),
    (None,
     'aria-label="Verified Unsloth"',
     "verified-publisher badge on Hub rows -- HF owner identity, not this app"),
    ("features/hub/catalog/owner-avatar.tsx",
     'name: "Unsloth"',
     "HF owner identity"),
    ("features/hub/catalog/owner-scope-toggle.tsx",
     'label: "Unsloth"',
     "HF owner identity"),
    ("features/hub/lib/channels.ts",
     "Unsloth Trending",
     "HF-publisher channel filter"),
    ("features/hub/lib/channels.ts",
     "published by Unsloth",
     "HF-publisher channel filter"),
    ("features/hub/lib/channels.ts",
     "Latest Unsloth",
     "HF-publisher channel filter"),
    ("features/hub/lib/channels.ts",
     "from the Unsloth channel",
     "HF-publisher channel filter"),
    ("features/hub/lib/channels.ts",
     "Latest Unsloth Models",
     "HF-publisher channel filter"),
    ("features/hub/lib/hub-token-header.ts",
     "X-Unsloth-HF-Token",
     "HTTP header name -- an API contract"),
    ("features/model-picker/components/model-selector/pickers.tsx",
     "Sort Unsloth models",
     "Hub \"Unsloth\" tab, an HF-publisher group"),
    ("features/model-picker/components/model-selector/pickers.tsx",
     "Search Unsloth models",
     "Hub \"Unsloth\" tab, an HF-publisher group"),
    ("features/model-picker/components/model-selector/pickers.tsx",
     "non-Unsloth models",
     "Hub \"Unsloth\" tab, an HF-publisher group"),
    ("features/model-picker/components/model-selector/pickers.tsx",
     "No matching Unsloth models.",
     "Hub \"Unsloth\" tab, an HF-publisher group"),
    ("features/model-picker/components/model-selector/pickers.tsx",
     "Unsloth",  # exact stripped line, see the equality check below
     "Hub \"Unsloth\" tab heading -- a bare JSX text node naming the HF-publisher group"),
    ("features/studio/preparation-progress.ts",
     "replace(/^Unsloth:",
     "strips a prefix the upstream training library emits"),
    ("features/studio/sections/training-memory-params.tsx",
     'value="unsloth">Unsloth<',
     "names the gradient-checkpointing technique \"unsloth\", not this product"),
]

# Lightweight comment tracker: `//` lines, and `/* ... */` / JSX `{/* ... */}`
# blocks (single- or multi-line). Not a full parser, but the frontend's actual
# comment styles are exactly these three, and false negatives here only make
# the check stricter (a missed comment still has to clear the allowlist).
def _is_comment_masked(lines):
    mask = [False] * len(lines)
    in_block = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if in_block:
            mask[i] = True
            if "*/" in line:
                in_block = False
            continue
        if not line:
            continue
        probe = line[1:] if line.startswith("{") else line
        if probe.startswith("/*"):
            mask[i] = True
            if "*/" not in probe[2:]:
                in_block = True
            continue
        if line.startswith("//"):
            mask[i] = True
    return mask

SPDX_LINE = re.compile(r"^\s*//\s*(SPDX-License-Identifier|Copyright)\b")

for path in sorted(FRONTEND.joinpath("src").rglob("*.ts*")):
    if "i18n" in path.parts and "locales" in path.parts:
        continue  # governed separately by checks #4 and #5
    text = path.read_text(encoding = "utf-8", errors = "replace")
    lines = text.split("\n")
    comment_masked = _is_comment_masked(lines)
    rel_to_src = path.relative_to(FRONTEND / "src").as_posix()
    for i, line in enumerate(lines):
        if SPDX_LINE.match(line) or comment_masked[i]:
            continue
        if not BRAND_WORD.search(line):
            continue
        allowed = False
        for file_suffix, substring, _reason in KEEP_ALLOWLIST:
            if file_suffix is not None and file_suffix != rel_to_src:
                continue
            # A bare "Unsloth" entry only allows a line that is *exactly* that
            # word once stripped (a bare JSX text node) -- never used as a
            # loose substring, which would also swallow unrelated hits in the
            # same file.
            if substring == "Unsloth":
                if line.strip() == "Unsloth":
                    allowed = True
                    break
                continue
            if substring in line:
                allowed = True
                break
        if not allowed:
            failures.append(
                f"{path.relative_to(ROOT)}:{i + 1}: leftover 'Unsloth' not on the "
                f"allowlist -- either rename it or add it to KEEP_ALLOWLIST with a "
                f"reason: {line.strip()[:160]!r}")

if failures:
    print("BRANDING CHECK FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("branding check passed")
