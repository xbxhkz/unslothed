// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Client for the draft-model chooser.
 *
 * This module deliberately does NOT build llama-server flags. The drafter flag
 * vocabulary is seven spellings across two families, each accepting `-f v` and
 * `-f=v`, with last-wins ordering; the backend already parses that set, and a
 * second implementation here would be free to drift from it. We send a choice
 * and receive a finished argument array.
 */

import { authFetch } from "@/features/auth";

export type DraftChoice = { kind: "local" | "hf"; ref: string } | null;

export type DraftCandidate = {
  kind: string;
  ref: string;
  label: string;
  source: "sidecar" | "local";
};

export type SelectResult = {
  ok: boolean;
  reason: string;
  detail: string;
  sizeBytes: number | null;
  vocabTarget: number | null;
  vocabDraft: number | null;
  llamaExtraArgs: string[] | null;
};

export async function fetchDraftCandidates(
  modelPath: string,
): Promise<DraftCandidate[]> {
  const url = `/api/draft-model/candidates?model_path=${encodeURIComponent(modelPath)}`;
  const res = await authFetch(url);
  if (!res.ok) {
    return [];
  }
  const body = await res.json().catch(() => null);
  return Array.isArray(body?.candidates) ? body.candidates : [];
}

export async function selectDraftModel(
  modelPath: string,
  existingArgs: string[],
  choice: DraftChoice,
): Promise<SelectResult> {
  const res = await authFetch("/api/draft-model/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_path: modelPath,
      // biome-ignore lint/style/useNamingConvention: API schema
      existing_args: existingArgs,
      choice: choice ? { kind: choice.kind, ref: choice.ref } : null,
    }),
  });
  const b = await res.json().catch(() => null);
  if (!res.ok || b === null) {
    // A non-2xx status or an unparsable body (dev-server HTML fallback, proxy
    // error page, a 500 with no JSON handler) is a real possibility, not a
    // theoretical one -- surface it as an ordinary rejection rather than
    // letting a thrown SyntaxError reach an unguarded caller and leave the
    // picker blank with no indication anything failed.
    return {
      ok: false,
      reason: "network",
      detail: `the backend could not be reached (HTTP ${res.status})`,
      sizeBytes: null,
      vocabTarget: null,
      vocabDraft: null,
      llamaExtraArgs: null,
    };
  }
  return {
    ok: Boolean(b?.ok),
    reason: String(b?.reason ?? ""),
    detail: String(b?.detail ?? ""),
    sizeBytes: b?.size_bytes ?? null,
    vocabTarget: b?.vocab_target ?? null,
    vocabDraft: b?.vocab_draft ?? null,
    llamaExtraArgs: Array.isArray(b?.llama_extra_args)
      ? b.llama_extra_args
      : null,
  };
}
