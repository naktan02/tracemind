import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "esbuild";

const appRoot = path.resolve(import.meta.dirname, "..", "..");
const tempDir = await mkdtemp(path.join(tmpdir(), "tracemind-collector-replay-"));
const testSource = path.join(tempDir, "collector-replay-unit.ts");
const testBundle = path.join(tempDir, "collector-replay-unit.mjs");

await writeFile(
  testSource,
  String.raw`
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";

import { SegmentBuffer } from "${appRoot}/src/collector/segmentBuffer.ts";

type ReplayEvent = {
  snapshotText: string;
  eventType: string;
  inputType: string | null;
  insertedText: string | null;
  isCompositionUpdate: boolean;
  advanceMs?: number;
  flushAfter?: boolean;
};

type ReplayCase = {
  name: string;
  expectedFinalTexts: Array<string | null>;
  expectedDeletedTexts?: Array<string | null>;
  events: ReplayEvent[];
};

type ReplayFixture = {
  schema_version: "collector_replay.v1";
  cases: ReplayCase[];
};

globalThis.window = {
  location: {
    origin: "https://blog.naver.com",
    href: "https://blog.naver.com/PostWriteForm.naver",
  },
  setTimeout: globalThis.setTimeout,
  clearTimeout: globalThis.clearTimeout,
} as unknown as Window & typeof globalThis;

const replayDir = path.join("${appRoot}", "tests", "collector", "replay-fixtures");
const fixtureFiles = readdirSync(replayDir)
  .filter((fileName) => fileName.endsWith(".json"))
  .sort();

const executedCases: string[] = [];

for (const fixtureFile of fixtureFiles) {
  const fixture = JSON.parse(
    readFileSync(path.join(replayDir, fixtureFile), "utf8"),
  ) as ReplayFixture;
  assert.equal(fixture.schema_version, "collector_replay.v1");
  for (const replayCase of fixture.cases) {
    runReplayCase(replayCase);
    executedCases.push(replayCase.name);
  }
}

function runReplayCase(replayCase: ReplayCase): void {
  const emitted: Array<{ final_text: string | null; deleted_text: string | null }> = [];
  const buffer = new SegmentBuffer(
    {
      idleMs: 5000,
      sourceType: "browser_extension",
    },
    (segment) =>
      emitted.push({
        final_text: segment.final_text,
        deleted_text: segment.deleted_text,
      }),
  );
  let nowMs = Date.parse("2026-06-02T16:44:51.735Z");
  for (const event of replayCase.events) {
    nowMs += event.advanceMs ?? 10;
    buffer.observe({
      elementId: "surface",
      snapshot: {
        text: event.snapshotText,
        surfaceType: "contenteditable",
        captureConfidence: "medium",
        fieldHint: null,
      },
      now: new Date(nowMs),
      eventType: event.eventType,
      inputType: event.inputType,
      insertedText: event.insertedText,
      isCompositionUpdate: event.isCompositionUpdate,
      locale: "ko",
    });
    if (event.flushAfter === true) {
      buffer.flushAll();
    }
  }
  buffer.flushAll();

  assert.deepEqual(
    emitted.map((segment) => segment.final_text),
    replayCase.expectedFinalTexts,
    replayCase.name,
  );
  if (replayCase.expectedDeletedTexts !== undefined) {
    assert.deepEqual(
      emitted.map((segment) => segment.deleted_text),
      replayCase.expectedDeletedTexts,
      replayCase.name,
    );
  }
}

console.log(
  JSON.stringify(
    {
      ok: true,
      fixture_count: fixtureFiles.length,
      case_count: executedCases.length,
      cases: executedCases,
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
