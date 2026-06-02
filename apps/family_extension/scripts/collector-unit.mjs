import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

const appRoot = path.resolve(import.meta.dirname, "..");
const tempDir = await mkdtemp(path.join(tmpdir(), "tracemind-collector-unit-"));
const testSource = path.join(tempDir, "collector-unit.ts");
const testBundle = path.join(tempDir, "collector-unit.mjs");

await writeFile(
  testSource,
  String.raw`
import assert from "node:assert/strict";

import { SegmentBuffer } from "${appRoot}/src/collector/segmentBuffer.ts";

globalThis.window = {
  location: {
    origin: "https://blog.naver.com",
    href: "https://blog.naver.com/PostWriteForm.naver",
  },
  setTimeout: globalThis.setTimeout,
  clearTimeout: globalThis.clearTimeout,
} as unknown as Window & typeof globalThis;

const baseContext = {
  elementId: "surface",
  snapshot: {
    surfaceType: "contenteditable" as const,
    captureConfidence: "medium" as const,
    fieldHint: null,
  },
  locale: "ko",
};

function createHarness(): {
  emitted: unknown[];
  observe: (
    text: string,
    eventType: string,
    inputType: string | null,
    insertedText: string | null,
    isCompositionUpdate: boolean,
  ) => void;
  advance: (ms: number) => void;
  flush: () => void;
} {
  const emitted: unknown[] = [];
  const buffer = new SegmentBuffer(
    {
      idleMs: 5000,
      sourceType: "browser_extension",
    },
    (segment) => emitted.push(segment),
  );
  let nowMs = Date.parse("2026-06-02T16:44:51.735Z");

  return {
    emitted,
    observe(
      text: string,
      eventType: string,
      inputType: string | null,
      insertedText: string | null,
      isCompositionUpdate: boolean,
    ): void {
      buffer.observe({
        ...baseContext,
        snapshot: {
          ...baseContext.snapshot,
          text,
        },
        now: new Date(nowMs),
        eventType,
        inputType,
        insertedText,
        isCompositionUpdate,
      });
      nowMs += 10;
    },
    advance(ms: number): void {
      nowMs += ms;
    },
    flush(): void {
      buffer.flushAll();
    },
  };
}

{
  const harness = createHarness();
  for (const part of ["난", " ", "너가", " ", "보고", " ", "싶다", " ", "아"]) {
    if (part === " ") {
      harness.observe("", "input", "insertText", part, false);
    } else {
      harness.observe(part, "compositionend", null, part, true);
    }
  }
  harness.observe("", "input", "deleteContentBackward", null, false);
  harness.observe("바다", "compositionend", null, "바다", true);
  harness.flush();

  assert.equal(harness.emitted.length, 1);
  const segment = harness.emitted[0] as {
    final_text: string | null;
    deleted_text: string | null;
  };

  assert.equal(segment.final_text, "난 너가 보고 싶다 바다");
  assert.equal(segment.deleted_text, "아");
}

{
  const harness = createHarness();
  harness.observe("대단하", "compositionend", null, "대단하", true);
  harness.observe("대단하네", "compositionend", null, "네", true);
  harness.observe("대단하", "input", "deleteContentBackward", null, false);
  harness.flush();

  assert.equal(harness.emitted.length, 1);
  const segment = harness.emitted[0] as {
    final_text: string | null;
    deleted_text: string | null;
  };

  assert.equal(segment.final_text, "대단하네");
  assert.equal(segment.deleted_text, null);
}

{
  const harness = createHarness();
  harness.observe("대단하", "compositionend", null, "대단하", true);
  harness.observe("대단하네", "compositionend", null, "네", true);
  harness.advance(500);
  harness.observe("대단하", "input", "deleteContentBackward", null, false);
  harness.flush();

  assert.equal(harness.emitted.length, 1);
  const segment = harness.emitted[0] as {
    final_text: string | null;
    deleted_text: string | null;
  };

  assert.equal(segment.final_text, "대단하");
  assert.equal(segment.deleted_text, "네");
}

{
  const harness = createHarness();
  harness.observe("첫 문장", "compositionend", null, "첫 문장", true);
  harness.flush();
  harness.advance(5000);
  harness.observe("첫 문장 다음 문장", "compositionend", null, " 다음 문장", true);
  harness.flush();

  assert.equal(harness.emitted.length, 2);
  const firstSegment = harness.emitted[0] as {
    final_text: string | null;
    deleted_text: string | null;
  };
  const secondSegment = harness.emitted[1] as {
    final_text: string | null;
    deleted_text: string | null;
  };

  assert.equal(firstSegment.final_text, "첫 문장");
  assert.equal(firstSegment.deleted_text, null);
  assert.equal(secondSegment.final_text, "다음 문장");
  assert.equal(secondSegment.deleted_text, null);
}

{
  const harness = createHarness();
  harness.observe("첫 줄", "compositionend", null, "첫 줄", true);
  harness.observe("첫 줄\n", "input", "insertParagraph", null, false);
  harness.flush();
  harness.advance(5000);
  harness.observe("첫 줄\n둘째 줄", "compositionend", null, "둘째 줄", true);
  harness.flush();

  assert.equal(harness.emitted.length, 2);
  const firstSegment = harness.emitted[0] as {
    final_text: string | null;
    deleted_text: string | null;
  };
  const secondSegment = harness.emitted[1] as {
    final_text: string | null;
    deleted_text: string | null;
  };

  assert.equal(firstSegment.final_text, "첫 줄");
  assert.equal(firstSegment.deleted_text, null);
  assert.equal(secondSegment.final_text, "둘째 줄");
  assert.equal(secondSegment.deleted_text, null);
}

{
  const harness = createHarness();
  harness.observe("모", "input", "insertCompositionText", "모", true);
  harness.observe("모르겠", "input", "insertCompositionText", "겠", true);
  harness.observe(
    "모르겠는 삶이다. 정말",
    "input",
    "insertCompositionText",
    "말",
    true,
  );
  harness.observe(
    "모르겠는 삶이다. 정말.",
    "input",
    "insertText",
    ".",
    false,
  );
  harness.flush();

  assert.equal(harness.emitted.length, 1);
  const segment = harness.emitted[0] as {
    final_text: string | null;
    deleted_text: string | null;
  };

  assert.equal(segment.final_text, "모르겠는 삶이다. 정말.");
  assert.equal(segment.deleted_text, null);
}

console.log(
  JSON.stringify(
    {
      ok: true,
      cases: [
        "composition-transcript",
        "phantom-ime-delete",
        "delayed-user-delete",
        "post-idle-baseline",
        "enter-linebreak-baseline",
        "composition-snapshot-beats-punctuation",
      ],
    },
    null,
    2,
  ),
);
`,
);

try {
  await build({
    entryPoints: [testSource],
    outfile: testBundle,
    bundle: true,
    format: "esm",
    platform: "node",
    target: "node22",
    logLevel: "silent",
  });
  await import(pathToFileURL(testBundle).href);
} finally {
  await rm(tempDir, { force: true, recursive: true });
}
