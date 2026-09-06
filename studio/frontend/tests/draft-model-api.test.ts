// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The draft-model chooser's client. Two properties matter more than the shapes:
//
//   1. A FAILED request must never come back looking like a successful empty
//      answer. The picker has three different things to say -- "no drafters
//      next to this model", "this model could not be resolved", and "the
//      backend did not answer" -- and the client is what keeps them apart.
//      `resolved: true` on a non-2xx collapsed all three into the first.
//   2. The pinned drafter is read back from the BACKEND, never parsed here.
//      The flag vocabulary is seven spellings in two value forms; a reader in
//      this file would be the second parser the module header forbids.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// draft-model-api reaches authFetch through the auth barrel, which re-exports
// login-page.tsx. See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);

type Reply = {
  status: number;
  body?: unknown;
  text?: string;
  throws?: boolean;
};

let reply: Reply = { status: 200, body: {} };
const requests: { url: string; init?: RequestInit }[] = [];

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  requests.push({ url: String(input), init });
  if (reply.throws) {
    // What authFetch does when nothing is listening: it throws rather than
    // resolving with a Response. See src/features/auth/api.ts.
    throw new TypeError("fetch failed");
  }
  return new Response(reply.text ?? JSON.stringify(reply.body ?? null), {
    status: reply.status,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const { fetchCurrentDraftPin, fetchDraftCandidates, selectDraftModel } =
  await import("../src/features/chat/api/draft-model-api.ts");

test("a successful candidates fetch reports both the list and resolution", async () => {
  reply = {
    status: 200,
    body: {
      candidates: [
        { kind: "local", ref: "/m/d.gguf", label: "d.gguf", source: "sidecar" },
      ],
      resolved: true,
    },
  };
  const r = await fetchDraftCandidates("/m/target.gguf", null);
  assert.equal(r.ok, true);
  assert.equal(r.resolved, true);
  assert.equal(r.candidates.length, 1);
});

test("an unresolvable model is resolved:false, not an empty success", async () => {
  reply = { status: 200, body: { candidates: [], resolved: false } };
  const r = await fetchDraftCandidates("org/never-cached", "Q4_K_M");
  assert.equal(r.ok, true, "the request itself succeeded");
  assert.equal(r.resolved, false);
});

test("a failed candidates fetch is not reported as a resolved empty list", async () => {
  // The defect: `return { candidates: [], resolved: true }` on !res.ok. Every
  // backend error, proxy error and dev-server HTML fallback then rendered as
  // "No colocated drafters found" -- a confident, wrong answer, and the third
  // site in this branch with that same shape.
  for (const status of [401, 404, 500, 502]) {
    reply = { status, body: { detail: "nope" } };
    const r = await fetchDraftCandidates("/m/target.gguf", null);
    assert.equal(
      r.ok,
      false,
      `HTTP ${status} must not read as a successful answer`,
    );
    assert.equal(
      r.resolved,
      false,
      `HTTP ${status} must not claim the backend resolved the model and found nothing`,
    );
    assert.deepEqual(r.candidates, []);
  }
});

test("an unparsable candidates body is a failure, not an empty list", async () => {
  // A dev-server HTML fallback answers 200 with a page, not JSON.
  reply = { status: 200, text: "<!doctype html><title>index</title>" };
  const r = await fetchDraftCandidates("/m/target.gguf", null);
  assert.equal(r.ok, false);
  assert.equal(r.resolved, false);
});

test("a thrown fetch propagates, so the caller must catch it", async () => {
  // Documented deliberately: authFetch throws before any Response exists, and
  // no guard in this module can see that. The picker's effect carries the
  // .catch() -- this asserts the contract those catches depend on.
  reply = { status: 200, throws: true };
  await assert.rejects(() => fetchDraftCandidates("/m/target.gguf", null));
});

test("the current pin is read from the backend, not parsed here", async () => {
  reply = { status: 200, body: { pin: { kind: "local", ref: "/m/d.gguf" } } };
  requests.length = 0;
  const r = await fetchCurrentDraftPin(["-md", "/m/d.gguf"]);
  assert.equal(r.ok, true);
  assert.deepEqual(r.pin, { kind: "local", ref: "/m/d.gguf" });
  // The arguments are sent verbatim: no spelling is interpreted client-side.
  assert.deepEqual(JSON.parse(String(requests[0]?.init?.body)), {
    // biome-ignore lint/style/useNamingConvention: API schema
    existing_args: ["-md", "/m/d.gguf"],
  });
});

test("no pin reads as null with ok:true", async () => {
  reply = { status: 200, body: { pin: null } };
  const r = await fetchCurrentDraftPin(["--threads", "8"]);
  assert.equal(r.ok, true);
  assert.equal(r.pin, null);
});

test("a failed pin read is ok:false, distinguishable from no pin", async () => {
  // Collapsing these is how a pinned model comes to render as "Automatic".
  reply = { status: 500, body: { detail: "boom" } };
  const r = await fetchCurrentDraftPin(["-md", "/m/d.gguf"]);
  assert.equal(r.ok, false);
  assert.equal(r.pin, null);
});

test("a malformed pin payload is treated as no pin, not as a pin on undefined", async () => {
  reply = { status: 200, body: { pin: { kind: "weird", ref: 7 } } };
  const r = await fetchCurrentDraftPin([]);
  assert.equal(r.ok, true);
  assert.equal(r.pin, null);
});

test("selectDraftModel surfaces a bad status as a rejection, not a throw", async () => {
  reply = { status: 503, text: "<html>gateway</html>" };
  const r = await selectDraftModel("/m/t.gguf", null, [], {
    kind: "local",
    ref: "/m/d.gguf",
  });
  assert.equal(r.ok, false);
  assert.equal(r.llamaExtraArgs, null);
  assert.match(r.detail, /503/);
});

test("an unverified hf verdict keeps its reason so the picker can say so", async () => {
  reply = {
    status: 200,
    body: {
      ok: true,
      reason: "unverified",
      detail: "not contacted",
      // biome-ignore lint/style/useNamingConvention: API schema
      size_bytes: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      vocab_target: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      vocab_draft: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      llama_extra_args: ["--spec-draft-hf", "org/drafter"],
    },
  };
  const r = await selectDraftModel("/m/t.gguf", null, [], {
    kind: "hf",
    ref: "org/drafter",
  });
  assert.equal(r.ok, true);
  assert.equal(r.reason, "unverified");
  assert.deepEqual(r.llamaExtraArgs, ["--spec-draft-hf", "org/drafter"]);
});
