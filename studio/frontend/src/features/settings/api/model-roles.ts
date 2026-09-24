// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { ApiRolesRead, RoleRow } from "../lib/model-roles-binding";
import { toRolesPayload } from "../lib/model-roles-binding";

export type ModelRolesRead = {
  rows: ApiRolesRead;
  /** The server's own sentence about stored-but-unapplied overrides. */
  overridesNote: string;
};

function readBody(body: ApiRolesRead): ModelRolesRead {
  return {
    rows: body,
    overridesNote:
      typeof body?.overrides_note === "string" ? body.overrides_note : "",
  };
}

/** GET /api/model-roles. Throws on a non-2xx, so the caller can tell a failure
 * from an empty set -- rendering those the same would be two opposite claims. */
export async function loadModelRoles(): Promise<ModelRolesRead> {
  const res = await authFetch("/api/model-roles");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, `Failed to load roles (${res.status})`));
  }
  return readBody((await res.json()) as ApiRolesRead);
}

/** PUT /api/model-roles. Replaces the whole set and returns the new state. */
export async function saveModelRoles(rows: RoleRow[]): Promise<ModelRolesRead> {
  const res = await authFetch("/api/model-roles", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ roles: toRolesPayload(rows) }),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, `Failed to save roles (${res.status})`));
  }
  return readBody((await res.json()) as ApiRolesRead);
}
