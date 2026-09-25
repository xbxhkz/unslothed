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
#
#    Case-INsensitive: a first version of this check matched "Unsloth" only,
#    which is a real blind spot -- two hardcoded `Logo()` components rendered
#    a lowercase `unsloth` wordmark (studio/frontend/src/components/tauri/
#    startup-screen.tsx and update-screen.tsx, shown on nearly every launch
#    and every update) and neither was on the original capitalized-only
#    allowlist, so the check passed while they sat unrenamed. Matching
#    case-insensitively catches that class of miss, at the cost of also
#    catching every lowercase functional string (CLI commands, CSS classes,
#    storage keys, HF repo-id prefixes, env vars) that must NOT be renamed --
#    hence the much larger allowlist below.
BRAND_WORD = re.compile(r"\bUnsloth\b(?!ed)", re.IGNORECASE)

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

    # -- Lowercase functional strings, surfaced by the case-insensitive sweep --
    (None,
     "unsloth studio update",
     "real CLI command; the `unsloth` package/binary is not renamed by this task"),
    (None,
     "unsloth studio reset-password",
     "real CLI command"),
    (None,
     "unsloth start",
     "real CLI command/subcommand"),
    ("features/auth/components/auth-form.tsx",
     'HIDDEN_LOGIN_USERNAME = "unsloth"',
     "the real default admin/login username (backend DEFAULT_ADMIN_USERNAME); "
     "renaming the literal would lock out the default account"),
    (None,
     "sk-unsloth-",
     "matches API_KEY_PREFIX in backend/auth/storage.py, checked with "
     "token.startswith(...); renaming misdocuments the real key format"),
    (None,
     "UNSLOTH_API_KEY",
     "real environment variable"),
    (None,
     "UNSLOTH_STUDIO_URL",
     "real environment variable read by `unsloth start`"),
    ("features/deep-links/parse-deep-link.ts",
     "unsloth://open_from_hf",
     "the registered deep-link URL scheme (matches tauri.conf.json's "
     "plugins.deep-link.desktop.schemes); renaming orphans existing links"),
    (None,
     "unslothai/unsloth",
     "the real GitHub path (github.com/unslothai/unsloth) for the upstream "
     "training library's source, issues, or license file -- not this product"),
    ("features/images/images-page.tsx",
     "Only unsloth or on-device image models can be loaded here",
     "describes the adjacent HF-repo-id gate (startsWith('unsloth/')) -- "
     "\"unsloth\" here is the HF publisher, matching the check right above it"),
    ("features/video/video-page.tsx",
     "Only unsloth or on-device video models can be loaded here",
     "describes the adjacent HF-repo-id gate (startsWith('unsloth/')) -- "
     "\"unsloth\" here is the HF publisher, matching the check right above it"),
    ("hooks/use-hardware-info.ts",
     "unsloth: string | null;",
     "field reporting the installed unsloth (training-library) package "
     "version from the backend's /versions endpoint -- a field name"),
    ("hooks/use-hardware-info.ts",
     "unsloth: null,",
     "same field, default value"),
    ("hooks/use-hardware-info.ts",
     "?.unsloth ?? null,",
     "same field, read from the backend response"),
    ("lib/tauri-diagnostics.ts",
     "unsloth\\/studio",
     "matches the real ~/.unsloth config directory this app reads/writes, "
     "used to redact local usernames out of shared diagnostics -- renaming "
     "the pattern without renaming the directory breaks the redaction "
     "(Unix path form)"),
    ("lib/tauri-diagnostics.ts",
     "unsloth\\\\studio",
     "same redaction regex, Windows path form (\\Users\\...\\.unsloth\\studio)"),
    ("features/profile/sloth-avatars.ts",
     "Sloth emojis/UnSloth",
     "matches the literal asset filename under public/Sloth emojis/; "
     "renaming the string without renaming the file 404s the image"),
    ("features/settings/components/remote-access-section.tsx",
     "sign in as unsloth",
     "the real default remote-login username (backend DEFAULT_ADMIN_USERNAME), "
     "named in a prose sentence -- matches the brief's own worked example: "
     "\"as unsloth\" stays, \"The Unsloth Desktop App\" (renamed) does not"),
]

# Broad structural patterns: a LOWERCASE "unsloth" immediately fused to a
# separator character it would never sit next to in English prose (prose
# always has a space: "the unsloth backend", never "theunsloth-backend").
# Each pattern is a whole *category* of hit, so it gets one reason for the
# category rather than one entry per occurrence -- there are ~300 such
# occurrences (HF repo ids alone account for over half of them, in the model
# catalog and its tests).
#
# Deliberately case-SENSITIVE (no re.IGNORECASE), unlike BRAND_WORD above.
# Every real identifier/URL/repo-id these patterns cover is conventionally
# all-lowercase (HF org slugs, CSS classes, storage keys, domains). Giving
# these IGNORECASE would let them also swallow a *capitalized* "Unsloth"
# sitting next to the same punctuation -- and a capitalized hit is exactly
# the shape of a real leftover brand mention. This was caught for real:
# control 2 below first used a case-insensitive `"unsloth"` pattern, which
# silently accepted `alt="Unsloth"` (a genuine un-renamed brand string) as
# though it were the lowercase HF-owner-slug case. Case-sensitivity here is
# load-bearing, not a style choice.
KEEP_PATTERNS = [
    (re.compile(r"unsloth/"),
     "the Hugging Face repo-id prefix \"unsloth/...\" (e.g. "
     "unsloth/Llama-3.1-8B-Instruct) naming a real model/dataset published "
     "by the unsloth HF org -- not a brand mention. Covers "
     ".startsWith(\"unsloth/\") gates too."),
    (re.compile(r"unsloth-[a-z0-9]"),
     "a lowercase, hyphen-joined \"unsloth-...\" token: a CSS class name, "
     "IndexedDB/Dexie database name, custom DOM event name, localStorage "
     "key, HTTP header name, quantization-suffix convention "
     "(-unsloth-bnb-4bit), or asset filename. English prose never "
     "concatenates the brand name directly against a hyphen."),
    (re.compile(r"unsloth:"),
     "a colon-namespaced custom event/message-channel name (e.g. "
     "\"unsloth:model-ejected\") or the registered deep-link URL scheme "
     "(\"unsloth:\" as url.protocol) -- an identifier, not prose"),
    (re.compile(r"unsloth\.[a-z]"),
     "a dot-namespaced localStorage/persistence key (e.g. "
     "\"unsloth.hub.modelsTab\") -- renaming drops every existing user's "
     "saved value for it on upgrade"),
    (re.compile(r'"unsloth"'),
     "a bare quoted literal \"unsloth\" -- either the Hugging Face "
     "owner/publisher slug (compared or assigned: owner ===/=== \"unsloth\", "
     "ownerScope: \"unsloth\") or the gradient-checkpointing technique name "
     "in a type union (\"none\" | \"true\" | \"unsloth\"); never brand prose, "
     "which is never a bare 7-character quoted token"),
    (re.compile(r"unsloth\.ai/"),
     "the real external site unsloth.ai (docs, install script, changelog) "
     "-- upstream's site, not this app"),
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


def _spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def _allowlist_spans(line, file_suffix_filter):
    """Every (start, end) span in `line` covered by a KEEP_ALLOWLIST entry
    that applies to this file. Computed per line, not per match: a match is
    "on the allowlist" only if one of these spans overlaps *its own* span --
    a covered span elsewhere on the line does not count.
    """
    spans = []
    for file_suffix, substring, _reason in KEEP_ALLOWLIST:
        if file_suffix is not None and file_suffix != file_suffix_filter:
            continue
        # A bare "Unsloth" entry only allows a line that is *exactly* that
        # word once stripped (a bare JSX text node): the whole line is the
        # match, so its span trivially covers itself.
        if substring == "Unsloth":
            if line.strip() == "Unsloth":
                start = line.index("Unsloth")
                spans.append((start, start + len("Unsloth")))
            continue
        start = 0
        while True:
            idx = line.find(substring, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(substring)))
            start = idx + 1
    return spans


def _pattern_spans(line):
    """Every (start, end) span in `line` covered by a KEEP_PATTERNS regex."""
    spans = []
    for pattern, _reason in KEEP_PATTERNS:
        for m in pattern.finditer(line):
            spans.append(m.span())
    return spans


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
        matches = list(BRAND_WORD.finditer(line))
        if not matches:
            continue
        # Computed once per line (cheap enough at this file count), then
        # checked per match below. What matters is that coverage is decided
        # per MATCH, not per line: a legitimate lowercase hit earlier on a
        # line must not excuse an unrelated capitalized leftover later on
        # the same line. An earlier version of this check tested only
        # `pattern.search(line)` / `substring in line` -- true for the whole
        # line the instant *any* occurrence matched anywhere on it -- which
        # is exactly the gap a composite line like
        #   `ownerScope === "unsloth" ? "Unsloth Verified" : "Other"`
        # slips through: the bare-quoted "unsloth" comparison is legitimate,
        # but "Unsloth Verified" is a real leftover, and line-scope matching
        # would have excused both.
        covering_spans = _allowlist_spans(line, rel_to_src) + _pattern_spans(line)
        for m in matches:
            start, end = m.span()
            if any(_spans_overlap(start, end, cs, ce) for cs, ce in covering_spans):
                continue
            failures.append(
                f"{path.relative_to(ROOT)}:{i + 1}:{start + 1}: leftover 'Unsloth' not "
                f"on the allowlist -- either rename it or add it to KEEP_ALLOWLIST/"
                f"KEEP_PATTERNS with a reason: {line.strip()[:160]!r}")

# 7. Tests that ASSERT on a renamed string. Checks 1-6 walk src/ only, so a
# rename can update a user-visible string and leave the test asserting the old
# brand. That is exactly what 90ae250b did: six tests went red and stayed red,
# plus a seventh that never went red at all --
#
#     source: "... is managed by Unslothed and cannot be passed here."
#     test:   assert.match(text, /managed by Unsloth/)
#
# which PASSED, because the renamed string CONTAINS the old one and the regex is
# unanchored. It read as a brand check while being true under either brand, so it
# could not detect a rename in either direction. A red test announces itself;
# that kind does not, and no test run will ever surface it.
#
# Deliberately narrow: only lines that ASSERT, not every mention. tests/ is full
# of legitimate "Unsloth" -- the HF org slug, "Unsloth/Repo-GGUF" repo ids,
# docs.unsloth.ai, and the LIBRARY's own log output fed in as fixture input
# ("Unsloth: Formatting dataset"). Those are DATA, and data does not go stale
# when the product is renamed; an assertion does. Flagging every mention would
# mean ~21 allowlist entries guarding one real check.
TEST_ASSERT = re.compile(r"\bassert\.\w+\(")
# Case-SENSITIVE, unlike BRAND_WORD, for the reason KEEP_PATTERNS gives above:
# the product name in a user-visible string is always capitalized, while every
# lowercase "unsloth" in tests is a functional token that is never renamed -- an
# HF repo id or cache path (models--unsloth--x), the `unsloth studio update` CLI
# command, a drag type (application/x-unsloth-...), a CSS class. Requiring the
# capital excludes all of those by construction rather than by allowlist, and a
# capitalized hit is exactly the shape of a real leftover.
TEST_BRAND = re.compile(r"\bUnsloth\b(?!ed)")
TESTS_KEEP = [
    ("model-row-owner.test.ts", 'isUnslothOwner("Unsloth")',
     "the Hugging Face ORG slug, which upstream owns and this fork does not "
     "rename -- the function under test decides HF ownership, not branding"),
    ("training-start-preparation.test.ts", "Unsloth: Formatting dataset",
     "a log line the Unsloth LIBRARY prints, fed in as parser input; the parser "
     "must keep matching what the library actually emits"),
]
tests_dir = FRONTEND / "tests"
if tests_dir.is_dir():
    for path in sorted(tests_dir.rglob("*.ts")):
        lines = path.read_text(encoding = "utf-8", errors = "replace").split("\n")
        comment_masked = _is_comment_masked(lines)
        for i, line in enumerate(lines):
            if SPDX_LINE.match(line) or comment_masked[i] or not TEST_ASSERT.search(line):
                continue
            if not TEST_BRAND.search(line):
                continue
            if any(name == path.name and frag in line for name, frag, _why in TESTS_KEEP):
                continue
            failures.append(
                f"{path.relative_to(ROOT)}:{i + 1}: a test ASSERTS on 'Unsloth'. If the "
                f"source says 'Unslothed', this assertion is stale -- and an unanchored "
                f"regex will pass on it as a substring rather than fail. Match the source "
                f"string exactly, or add it to TESTS_KEEP with a reason: "
                f"{line.strip()[:160]!r}")

if failures:
    print("BRANDING CHECK FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("branding check passed")
