// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the Model roles section decides, separated from how it renders: the
// component imports React and cannot be loaded by the node --test suite, and
// these three decisions are the ones worth pinning.

/** Three-state readiness, as core/inference/tool_readiness reports it. */
export type RoleState = "ready" | "missing" | "unknown";

export type RoleRow = {
  role: string;
  /** "" when the role is not bound. */
  model: string;
  state: RoleState;
  detail: string;
};

/** GET /api/model-roles, in the shape routes/model_roles.py returns it. */
export type ApiRolesRead = {
  roles?: Record<
    string,
    // `overrides` and `overrides_applied` are returned and deliberately unused
    // here; see toRolesPayload for why they are never sent back.
    | {
        model?: unknown;
        state?: unknown;
        detail?: unknown;
        overrides?: unknown;
        // biome-ignore lint/style/useNamingConvention: API schema
        overrides_applied?: unknown;
      }
    | undefined
  >;
  defaults?: unknown;
  // biome-ignore lint/style/useNamingConvention: API schema
  overrides_note?: unknown;
};

/**
 * The roles `ask_model` will actually accept.
 *
 * `image` and `video` are bindable and are deliberately NOT here: their bindings
 * record a preference and feed defaults, while a delegation loads onto the CHAT
 * backend -- so delegation refuses them rather than unloading the user's chat
 * model for something that cannot answer a task. The section says so per row
 * instead of leaving the user to discover it from a refusal.
 */
export const DELEGATABLE_ROLES = ["primary", "coding", "vision"] as const;

function asState(value: unknown): RoleState {
  return value === "ready" || value === "missing" ? value : "unknown";
}

function asText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/**
 * One row per role: every role the server calls a default, plus any bound role
 * it does not.
 *
 * The second half matters more than it looks. PUT replaces the WHOLE set, so a
 * bound role missing from the rows is a binding the next save silently deletes.
 * Keeping it visible is what stops the UI destroying a binding it never showed.
 */
export function mergeRoleRows(read: ApiRolesRead | null): RoleRow[] {
  const roles = read?.roles && typeof read.roles === "object" ? read.roles : {};
  const defaults = Array.isArray(read?.defaults)
    ? read.defaults.filter((name): name is string => typeof name === "string")
    : [];
  const names = [...defaults];
  for (const name of Object.keys(roles)) {
    if (!names.includes(name)) {
      names.push(name);
    }
  }
  return names.map((role) => {
    const entry = roles[role];
    return {
      role,
      model: asText(entry?.model),
      state: asState(entry?.state),
      detail: asText(entry?.detail),
    };
  });
}

/**
 * The body for PUT /api/model-roles.
 *
 * Unbound roles are omitted rather than sent with an empty model: the endpoint
 * rejects a binding whose model is blank (400, "a model is required"), so
 * sending the empty rows this section always renders would make every save fail.
 *
 * `overrides` is never sent. The API stores and echoes it and applies it to
 * nothing, so round-tripping it here would make the section look like an
 * overrides editor and re-persist values on every save.
 */
export function toRolesPayload(rows: RoleRow[]): Record<string, { model: string }> {
  const out: Record<string, { model: string }> = {};
  for (const row of rows) {
    const model = row.model.trim();
    if (model) {
      out[row.role] = { model };
    }
  }
  return out;
}
