// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The draft-model picker's READ-BACK half, which typecheck, lint and build all
// pass without exercising at all.
//
// Two properties, both of which were broken and neither of which any other test
// in this branch could see:
//
//   1. A model that is already pinned must render as pinned. The control is the
//      spec's named mitigation for the raw-args box and the picker sharing one
//      field ("the chooser reflecting whatever is already there"); rendering
//      "Automatic" over a real pin is the failure that mitigation exists to
//      prevent. It also has to work for arguments that arrive AFTER mount,
//      which is the common path: llama_extra_args hydrates asynchronously.
//   2. A failed candidates request must not render as "No colocated drafters
//      found". That is the one message the backend grew a `resolved` flag to
//      keep separate from a real answer.
//
// The component is driven directly, with a React whose renders the test
// controls, the same cut tests/personalization-locale-hydration.test.ts makes.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

const PICKER_URL = new URL(
  "../src/features/chat/components/draft-model-picker.tsx",
  import.meta.url,
);

type Slot = {
  value?: unknown;
  deps?: readonly unknown[];
  cleanup?: unknown;
  set?: boolean;
};

function runCleanup(cleanup: unknown): void {
  if (typeof cleanup === "function") (cleanup as () => void)();
}

function sameDeps(a: readonly unknown[], b: readonly unknown[]): boolean {
  return a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
}

/** The four hooks the picker uses, with renders and effects the test drives. */
function createReact() {
  const slots: Slot[] = [];
  const effects: (() => void)[] = [];
  let cursor = 0;
  let dirty = false;

  const slot = (): Slot => {
    const existing = slots[cursor];
    if (existing) {
      cursor += 1;
      return existing;
    }
    const created: Slot = {};
    slots[cursor] = created;
    cursor += 1;
    return created;
  };

  const react = {
    useState<T>(initial: T): [T, (next: T) => void] {
      const self = slot();
      if (!self.set) {
        self.value = initial;
        self.set = true;
      }
      return [
        self.value as T,
        (next: T) => {
          if (Object.is(next, self.value)) return;
          self.value = next;
          dirty = true;
        },
      ];
    },
    useRef<T>(initial: T): { current: T } {
      const self = slot();
      if (!self.set) {
        self.value = { current: initial };
        self.set = true;
      }
      return self.value as { current: T };
    },
    useCallback<T>(fn: T, deps: readonly unknown[]): T {
      const self = slot();
      if (!self.deps || !sameDeps(self.deps, deps)) {
        self.value = fn;
        self.deps = deps;
      }
      return self.value as T;
    },
    useEffect(fn: () => unknown, deps: readonly unknown[]): void {
      const self = slot();
      if (self.deps && sameDeps(self.deps, deps)) return;
      self.deps = deps;
      effects.push(() => {
        runCleanup(self.cleanup);
        self.cleanup = fn();
      });
    },
  };

  return {
    react,
    /** Renders until no state change is left, running effects after each pass. */
    flush<T>(body: () => T): T {
      let last!: T;
      do {
        cursor = 0;
        dirty = false;
        last = body();
        while (effects.length) effects.shift()?.();
      } while (dirty);
      return last;
    },
  };
}

/** Lets every pending promise callback run. */
function settle(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/** Every element in a rendered tree, depth first. */
function* walk(node: unknown): Generator<StubElement> {
  if (node === null || node === undefined || typeof node !== "object") return;
  if (Array.isArray(node)) {
    for (const child of node) yield* walk(child);
    return;
  }
  const element = node as StubElement;
  if (!("props" in element)) return;
  yield element;
  yield* walk(element.props?.children);
}

function findByType(tree: unknown, type: string): StubElement[] {
  return [...walk(tree)].filter((element) => element.type === type);
}

/** Every string of visible copy in the tree. */
function textOf(tree: unknown): string {
  const out: string[] = [];
  const push = (node: unknown): void => {
    if (typeof node === "string") {
      out.push(node);
      return;
    }
    if (Array.isArray(node)) {
      for (const child of node) push(child);
      return;
    }
    if (node && typeof node === "object" && "props" in (node as StubElement)) {
      push((node as StubElement).props?.children);
    }
  };
  push(tree);
  return out.join(" ");
}

const UI_SELECT = {
  Select: "Select",
  SelectContent: "SelectContent",
  SelectItem: "SelectItem",
  SelectTrigger: "SelectTrigger",
  SelectValue: "SelectValue",
};

type Candidates = { candidates: unknown[]; resolved: boolean; ok: boolean };
type Pin = { pin: { kind: string; ref: string } | null; ok: boolean };

type World = {
  candidates?: Candidates;
  candidatesThrows?: boolean;
  pin?: Pin;
  select?: Record<string, unknown>;
  args?: string[];
  hydrating?: boolean;
  speculativeType?: string;
};

function setup(world: World) {
  const host = createReact();
  const pinRequests: string[][] = [];
  const selectRequests: unknown[] = [];
  const written: string[][] = [];

  const api = {
    // biome-ignore lint/style/useNamingConvention: mirrors the exported constant
    VERDICT_UNVERIFIED: "unverified",
    fetchDraftCandidates: async () => {
      if (world.candidatesThrows) {
        // authFetch throws outright when the backend is not listening.
        throw new TypeError("fetch failed");
      }
      return world.candidates ?? { candidates: [], resolved: true, ok: true };
    },
    fetchCurrentDraftPin: async (existingArgs: string[]) => {
      pinRequests.push(existingArgs);
      return world.pin ?? { pin: null, ok: true };
    },
    selectDraftModel: async (
      _modelId: string,
      _variant: string | null,
      _existing: string[],
      choice: unknown,
    ) => {
      selectRequests.push(choice);
      return (
        world.select ?? {
          ok: true,
          reason: "ok",
          detail: "",
          sizeBytes: null,
          vocabTarget: null,
          vocabDraft: null,
          llamaExtraArgs: [],
        }
      );
    },
  };

  const { DraftModelPicker } = loadWithStubs<{
    DraftModelPicker: (props: Record<string, unknown>) => unknown;
  }>(PICKER_URL, {
    react: host.react,
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/button": { Button: "Button" },
    "@/components/ui/info-hint": { InfoHint: "InfoHint" },
    "@/components/ui/input": { Input: "Input" },
    "@/components/ui/select": UI_SELECT,
    // Echoes the key back rather than a translated string: these tests assert
    // on the note's presence/absence, not its copy, so the key alone binds
    // the assertion to the same string the component actually looked up.
    "@/i18n": { useT: () => (key: string) => key },
    "../api/draft-model-api": api,
  });

  let args = world.args ?? [];
  let tree: unknown = null;

  const render = () => {
    tree = host.flush(() =>
      DraftModelPicker({
        modelId: "/m/target.gguf",
        ggufVariant: null,
        speculativeType: world.speculativeType ?? "auto",
        existingArgs: args,
        hydrating: world.hydrating ?? false,
        onArgsChange: (next: string[]) => {
          written.push(next);
          args = next;
        },
      }),
    );
    return tree;
  };

  return {
    pinRequests,
    selectRequests,
    written,
    /** Mutable: a test can change what the backend answers mid-life, which is
     *  the only way to model arguments hydrating into a MOUNTED picker. */
    world,
    render,
    setArgs(next: string[]) {
      args = next;
    },
    setHydrating(next: boolean) {
      world.hydrating = next;
    },
    get tree() {
      return tree;
    },
  };
}

test("a pinned model renders as pinned, not as Automatic", async () => {
  // C1. The picker never seeded its controls from existingArgs, so a model with
  // --model-draft already saved opened showing "Automatic" -- over a pin the
  // user could neither see nor knowingly replace.
  const app = setup({
    args: ["--threads", "8", "--model-draft", "/m/d.gguf"],
    pin: { pin: { kind: "local", ref: "/m/d.gguf" }, ok: true },
    candidates: {
      candidates: [
        { kind: "local", ref: "/m/d.gguf", label: "d.gguf", source: "sidecar" },
      ],
      resolved: true,
      ok: true,
    },
  });

  app.render();
  await settle();
  await settle();
  const tree = app.render();

  const [select] = findByType(tree, "Select");
  assert.ok(select, "the Select must render");
  assert.equal(
    select.props.value,
    "/m/d.gguf",
    "the pinned drafter must be the selected value, not the Automatic sentinel",
  );
});

test("a pin arriving after mount is picked up, not missed", async () => {
  // C2(a). llama_extra_args hydrates asynchronously; with Advanced already open
  // from localStorage the picker mounts with `[]`. A mount-only effect never
  // ran again, so the pin was invisible on every visit after the first.
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
  });

  app.render();
  await settle();
  app.render();
  const [before] = findByType(app.tree, "Select");
  assert.equal(before?.props.value, "__automatic__");

  // Hydration lands on the ALREADY MOUNTED picker: same instance, new args.
  app.pinRequests.length = 0;
  app.world.pin = { pin: { kind: "local", ref: "/m/late.gguf" }, ok: true };
  app.setArgs(["-md", "/m/late.gguf"]);
  app.render();
  await settle();
  await settle();
  const tree = app.render();

  assert.ok(
    app.pinRequests.length > 0,
    "the picker must re-read the pin when the arguments change, not only on mount",
  );
  assert.deepEqual(
    app.pinRequests.at(-1),
    ["-md", "/m/late.gguf"],
    "and it must send the arguments that actually arrived",
  );
  const [after] = findByType(tree, "Select");
  assert.equal(
    after?.props.value,
    "/m/late.gguf",
    "a pin that arrives after mount must reach the control, not just the request",
  );
});

test("the rare spellings reach the backend rather than being parsed here", async () => {
  // C2(b). The old inline reader matched only `--model-draft` and
  // `--spec-draft-hf`; these five plus the =value and underscore forms all read
  // as "no pin". None of them is interpreted in this file any more, so the
  // assertion is that each is sent verbatim for the backend to answer.
  for (const args of [
    ["-md", "/d.gguf"],
    ["--spec-draft-model", "/d.gguf"],
    ["-hfd", "a/b"],
    ["-hfrd", "a/b"],
    ["--hf-repo-draft", "a/b"],
    ["--model-draft=/d.gguf"],
    ["--model_draft", "/d.gguf"],
  ]) {
    const app = setup({ args, pin: { pin: null, ok: true } });
    app.render();
    await settle();
    app.render();
    assert.deepEqual(
      app.pinRequests[0],
      args,
      `${args[0]} must be sent to the one parser, not interpreted locally`,
    );
  }
});

test("an hf pin seeds the repo field so the user can see what is pinned", async () => {
  const app = setup({
    args: ["--spec-draft-hf", "org/drafter"],
    pin: { pin: { kind: "hf", ref: "org/drafter" }, ok: true },
    select: {
      ok: true,
      reason: "unverified",
      detail: "not contacted",
      sizeBytes: null,
      vocabTarget: null,
      vocabDraft: null,
      llamaExtraArgs: ["--spec-draft-hf", "org/drafter"],
    },
  });

  app.render();
  await settle();
  await settle();
  const tree = app.render();

  const [input] = findByType(tree, "Input");
  assert.equal(input?.props.value, "org/drafter");
});

test("an unverified hf verdict is reported, not left as silence", async () => {
  // I2. size and vocabulary are both null for a repo, so the note stayed empty
  // and an hf pick produced no user-visible feedback at all -- an affirmative
  // "ok" the user never saw, for a choice nothing had checked.
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
    select: {
      ok: true,
      reason: "unverified",
      detail: "not contacted",
      sizeBytes: null,
      vocabTarget: null,
      vocabDraft: null,
      llamaExtraArgs: ["--spec-draft-hf", "org/drafter"],
    },
  });
  app.render();
  await settle();
  app.render();

  const [button] = findByType(app.tree, "Button").filter(
    (b) =>
      typeof b.props.onClick === "function" && b.props.children === "Use repo",
  );
  assert.ok(button, "the Use repo button must render");
  (button.props.onClick as () => void)();
  await settle();
  const tree = app.render();

  assert.match(
    textOf(tree),
    /could not be verified/i,
    "an unverified pin must say so; an affirmative ok with no note is the silent pass",
  );
});

test("a failed candidates fetch does not render as 'no drafters'", async () => {
  // I1. `resolved: true` on !res.ok turned every backend error into the one
  // message the design went out of its way to make distinguishable.
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
    candidates: { candidates: [], resolved: false, ok: false },
  });
  app.render();
  await settle();
  const tree = app.render();

  const text = textOf(tree);
  assert.doesNotMatch(
    text,
    /No colocated drafters found/,
    "a request that failed must not be reported as a model that has no drafters",
  );
  assert.match(text, /Could not reach the backend/i);
});

test("a thrown candidates fetch does not render as 'no drafters' either", async () => {
  // The missing .catch(): authFetch throws before any Response exists, so the
  // client's own guards cannot see it. Unhandled, the list stayed empty and
  // rendered the same confident wrong answer.
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
    candidatesThrows: true,
  });
  app.render();
  await settle();
  const tree = app.render();

  const text = textOf(tree);
  assert.doesNotMatch(text, /No colocated drafters found/);
  assert.match(text, /Could not reach the backend/i);
});

test("an unresolvable model still reads as unresolvable, not unreachable", async () => {
  // The three states stay three: this is the one the branch already had, and
  // it must survive the new one being added beside it.
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
    candidates: { candidates: [], resolved: false, ok: true },
  });
  app.render();
  await settle();
  const tree = app.render();
  assert.match(textOf(tree), /Could not resolve this model/i);
});

test("a genuinely empty answer still says no drafters were found", async () => {
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
    candidates: { candidates: [], resolved: true, ok: true },
  });
  app.render();
  await settle();
  const tree = app.render();
  assert.match(textOf(tree), /No colocated drafters found/);
});

test("the picker is inert while the stored arguments are still hydrating", async () => {
  // I5. In that window config.llamaExtraArgs is still `undefined`, which is
  // exactly the signal the page's hydration guards read to decide the user has
  // typed. A pick landing first makes hydration discard the stored list.
  const app = setup({
    args: [],
    hydrating: true,
    pin: { pin: null, ok: true },
  });
  app.render();
  await settle();
  const tree = app.render();

  const [select] = findByType(tree, "Select");
  assert.equal(
    select?.props.disabled,
    true,
    "the dropdown must not be clickable yet",
  );
  const [input] = findByType(tree, "Input");
  assert.equal(input?.props.disabled, true);
  for (const button of findByType(tree, "Button")) {
    assert.equal(
      button.props.disabled,
      true,
      "no button may write during hydration",
    );
  }
  assert.equal(
    app.pinRequests.length,
    0,
    "and nothing is read back against the placeholder argument list",
  );
});

test("a pin that is not among the candidates is still shown as selected", async () => {
  // A hand-written --model-draft, or one the candidates request could not
  // list. Radix renders an empty trigger for a value with no matching item,
  // which looks exactly like Automatic.
  const app = setup({
    args: ["--model-draft", "/elsewhere/hand-written.gguf"],
    pin: {
      pin: { kind: "local", ref: "/elsewhere/hand-written.gguf" },
      ok: true,
    },
    candidates: { candidates: [], resolved: true, ok: true },
  });
  app.render();
  await settle();
  await settle();
  const tree = app.render();

  const [select] = findByType(tree, "Select");
  assert.equal(select?.props.value, "/elsewhere/hand-written.gguf");
  const items = findByType(tree, "SelectItem").map((i) => i.props.value);
  assert.ok(
    items.includes("/elsewhere/hand-written.gguf"),
    "an item must exist for the pinned value or the trigger renders blank",
  );
});

test("a failed pin read is not reported as Automatic", async () => {
  const app = setup({
    args: ["-md", "/m/d.gguf"],
    pin: { pin: null, ok: false },
  });
  app.render();
  await settle();
  const tree = app.render();
  assert.match(
    textOf(tree),
    /could not reach the backend to read the pinned drafter/i,
  );
});

test("the picker does not re-read the pin for its own write", async () => {
  // The mount-only dependency list was trying to avoid re-validating our own
  // writes. Keying on argument CONTENT keeps that property without going blind
  // to hydration.
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
    candidates: {
      candidates: [
        { kind: "local", ref: "/m/d.gguf", label: "d.gguf", source: "local" },
      ],
      resolved: true,
      ok: true,
    },
    select: {
      ok: true,
      reason: "ok",
      detail: "d.gguf",
      sizeBytes: 1024,
      vocabTarget: 32,
      vocabDraft: 32,
      llamaExtraArgs: ["--model-draft", "/m/d.gguf"],
    },
  });
  app.render();
  await settle();
  app.render();
  const before = app.pinRequests.length;

  const [select] = findByType(app.tree, "Select");
  (select?.props.onValueChange as (v: string) => void)("/m/d.gguf");
  await settle();
  app.render();
  await settle();
  app.render();

  assert.deepEqual(app.written.at(-1), ["--model-draft", "/m/d.gguf"]);
  assert.equal(
    app.pinRequests.length,
    before,
    "our own write must not trigger a re-read and a second verdict",
  );
});

const AUTO_LOAD_NOTE_KEY = "chat.draftModelPicker.autoLoadFallbackNote";

test("the auto-load fallback note appears when a pin and a forced speculative type coexist", async () => {
  // I3 (routes/draft_model.py's module docstring): a pin survives a Run
  // Settings load, which always sends llama_extra_args explicitly, but is
  // silently stripped by the inherited-extras path (an auto-switch load, an
  // idle reload, or a chat-settings Apply) whenever that load also carries an
  // explicit, non-auto speculative_type. Both conditions have to hold for the
  // note to be worth showing.
  const app = setup({
    args: ["--model-draft", "/m/d.gguf"],
    pin: { pin: { kind: "local", ref: "/m/d.gguf" }, ok: true },
    speculativeType: "mtp",
  });
  app.render();
  await settle();
  await settle();
  const tree = app.render();

  assert.match(
    textOf(tree),
    new RegExp(AUTO_LOAD_NOTE_KEY.replace(/[.]/g, "\\.")),
    "a pin plus a forced (non-auto) speculative type must surface the note",
  );
});

test("the auto-load fallback note does not appear without a pin", async () => {
  const app = setup({
    args: [],
    pin: { pin: null, ok: true },
    speculativeType: "mtp",
  });
  app.render();
  await settle();
  await settle();
  const tree = app.render();

  assert.doesNotMatch(
    textOf(tree),
    new RegExp(AUTO_LOAD_NOTE_KEY.replace(/[.]/g, "\\.")),
    "no pin means nothing for an inherited load to strip -- the note has nothing to warn about",
  );
});

test("the auto-load fallback note does not appear when the speculative type is auto", async () => {
  // strip_spec (openai_auto_switch_settings.py / routes/inference.py) only
  // turns on when the saved config sets speculative_type explicitly.
  // "auto"/unset is stored as null and never reaches fields_set, so an
  // inherited load keeps the pin -- there is nothing this model needs the
  // workaround for.
  const app = setup({
    args: ["--model-draft", "/m/d.gguf"],
    pin: { pin: { kind: "local", ref: "/m/d.gguf" }, ok: true },
    speculativeType: "auto",
  });
  app.render();
  await settle();
  await settle();
  const tree = app.render();

  assert.doesNotMatch(
    textOf(tree),
    new RegExp(AUTO_LOAD_NOTE_KEY.replace(/[.]/g, "\\.")),
    "auto never sets an explicit speculative_type to inherit, so the pin is never stripped",
  );
});
