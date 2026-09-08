// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export interface WorkspaceRootWarning {
  code: string;
  message: string;
}

export interface WorkspaceRootState {
  path: string | null;
  warnings: WorkspaceRootWarning[];
}

export interface WorkspaceRootPreview extends WorkspaceRootState {
  exists: boolean;
}

function normalizeWarnings(value: unknown): WorkspaceRootWarning[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is WorkspaceRootWarning =>
      !!item &&
      typeof item === "object" &&
      typeof (item as WorkspaceRootWarning).code === "string" &&
      typeof (item as WorkspaceRootWarning).message === "string",
  );
}

// Every call here guards its own `.json()` parse and never throws: `authFetch`
// throws on transport failure, and an unhandled rejection from any of these
// would leave the field blank with no explanation instead of a message the
// caller can show. `null` is the single "something went wrong" signal.

/** The global workspace root: where chats outside a project work when they
 * are not sandboxed. */
export async function loadWorkspaceRoot(): Promise<WorkspaceRootState | null> {
  try {
    const response = await authFetch("/api/settings/workspace-root");
    const body = await response.json().catch(() => null);
    if (!response.ok || !body) return null;
    return {
      path: body.path ?? null,
      warnings: normalizeWarnings(body.warnings),
    };
  } catch {
    return null;
  }
}

export async function saveWorkspaceRoot(
  path: string | null,
): Promise<WorkspaceRootState | null> {
  try {
    const response = await authFetch("/api/settings/workspace-root", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok || !body) return null;
    return {
      path: body.path ?? null,
      warnings: normalizeWarnings(body.warnings),
    };
  } catch {
    return null;
  }
}

/**
 * Classify a candidate path WITHOUT storing it, so a warning can be shown
 * before the user commits rather than after. Shared by the global setting
 * and a project's own folder field -- classification has nothing specific
 * to what is being edited, only to the path itself.
 */
export async function previewWorkspaceRoot(
  path: string,
): Promise<WorkspaceRootPreview | null> {
  try {
    const response = await authFetch("/api/settings/workspace-root/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok || !body) return null;
    return {
      path: body.path ?? null,
      warnings: normalizeWarnings(body.warnings),
      exists: Boolean(body.exists),
    };
  } catch {
    return null;
  }
}
