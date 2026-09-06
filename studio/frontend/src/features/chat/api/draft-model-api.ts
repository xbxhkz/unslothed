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

/**
 * `resolved` is separate from an empty `candidates` list on purpose: the backend
 * resolves `modelId` (an HF repo id or a local path) to an actual local GGUF file
 * before it can glob that file's directory for siblings, and that resolution can
 * fail (nothing cached locally under this id/variant yet). An empty list for that
 * reason reads as "no drafters were found" when the truth is "this model could not
 * even be checked" -- two different things the picker has to say differently.
 */
export type DraftCandidatesResult = {
  candidates: DraftCandidate[];
  resolved: boolean;
};

export async function fetchDraftCandidates(
  modelId: string,
  ggufVariant: string | null,
): Promise<DraftCandidatesResult> {
  const params = new URLSearchParams({
    // biome-ignore lint/style/useNamingConvention: API schema
    model_id: modelId,
  });
  if (ggufVariant) {
    params.set("gguf_variant", ggufVariant);
  }
  const url = `/api/draft-model/candidates?${params.toString()}`;
  const res = await authFetch(url);
  if (!res.ok) {
    return { candidates: [], resolved: true };
  }
  const body = await res.json().catch(() => null);
  return {
    candidates: Array.isArray(body?.candidates) ? body.candidates : [],
    resolved: Boolean(body?.resolved),
  };
}

export async function selectDraftModel(
  modelId: string,
  ggufVariant: string | null,
  existingArgs: string[],
  choice: DraftChoice,
): Promise<SelectResult> {
  const res = await authFetch("/api/draft-model/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelId,
      // biome-ignore lint/style/useNamingConvention: API schema
      gguf_variant: ggufVariant,
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
