// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The logic behind the Model roles settings section.
 *
 * model-roles-section.tsx imports React, so it cannot be loaded here. The three
 * decisions worth pinning live in a plain module, which this drives directly.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  DELEGATABLE_ROLES,
  type RoleRow,
  mergeRoleRows,
  toRolesPayload,
} from "../src/features/settings/lib/model-roles-binding.ts";

const READ = {
  roles: {
    coding: {
      model: "repo/Coder",
      overrides: {},
      // biome-ignore lint/style/useNamingConvention: API schema
      overrides_applied: false,
      state: "ready",
      detail: "repo/Coder is downloaded",
    },
  },
  defaults: ["primary", "coding", "vision", "image", "video"],
  // biome-ignore lint/style/useNamingConvention: API schema
  overrides_note: "Role overrides are stored and returned but are NOT applied.",
};

test("every default role gets a row, bound or not", () => {
  const rows = mergeRoleRows(READ);
  assert.deepEqual(
    rows.map((r) => r.role),
    ["primary", "coding", "vision", "image", "video"],
    "an unbound role must still be visible, or you cannot discover it is bindable",
  );
  const coding = rows.find((r) => r.role === "coding");
  assert.equal(coding?.model, "repo/Coder");
  assert.equal(coding?.state, "ready");
  assert.equal(rows.find((r) => r.role === "vision")?.model, "");
});

test("a bound role the server does not list as a default is kept, not dropped", () => {
  const rows = mergeRoleRows({
    ...READ,
    roles: { ...READ.roles, research: { model: "repo/R", overrides: {}, state: "unknown" } },
  });
  assert.ok(
    rows.some((r) => r.role === "research" && r.model === "repo/R"),
    "the PUT replaces the whole set, so a row we drop is a binding we silently delete",
  );
});

test("the payload omits unbound roles rather than sending them empty", () => {
  const rows = mergeRoleRows(READ);
  const payload = toRolesPayload(rows);
  assert.deepEqual(Object.keys(payload), ["coding"]);
  assert.deepEqual(payload.coding, { model: "repo/Coder" });
});

test("a model is trimmed, and whitespace alone counts as unbound", () => {
  const rows: RoleRow[] = [
    { role: "coding", model: "  repo/Coder  ", state: "ready", detail: "" },
    { role: "vision", model: "   ", state: "unknown", detail: "" },
  ];
  const payload = toRolesPayload(rows);
  assert.deepEqual(payload, { coding: { model: "repo/Coder" } });
});

test("overrides are never sent back, because nothing reads them", () => {
  // The API stores and echoes `overrides` and applies them to nothing. Echoing
  // them back on save would make the section look like an overrides editor and
  // would re-persist values a future version might start honouring.
  const rows = mergeRoleRows(READ);
  const payload = toRolesPayload(rows);
  assert.equal("overrides" in payload.coding, false);
});

test("image and video are not delegatable, and the list says which are", () => {
  // ask_model refuses these two: their bindings record a preference and the chat
  // backend is what a delegation loads onto. The UI must not imply otherwise.
  assert.deepEqual(DELEGATABLE_ROLES, ["primary", "coding", "vision"]);
});
