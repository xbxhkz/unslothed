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

if failures:
    print("BRANDING CHECK FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("branding check passed")
