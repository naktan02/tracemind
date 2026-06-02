import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { build } from "esbuild";
import { chromium } from "playwright";

const appRoot = path.resolve(import.meta.dirname, "..");
const tempDir = await mkdtemp(path.join(tmpdir(), "tracemind-surface-unit-"));
const entrySource = path.join(tempDir, "surface-detector-entry.ts");
const browserBundle = path.join(tempDir, "surface-detector-entry.js");
const chromeExecutablePath =
  process.env.PLAYWRIGHT_CHROME_EXECUTABLE ?? "/usr/bin/google-chrome";

await writeFile(
  entrySource,
  String.raw`
import { readTextSurfaceSnapshot } from "${appRoot}/src/collector/surfaceDetector.ts";

globalThis.readSnapshotForSelector = (selector: string) => {
  const result = readTextSurfaceSnapshot(document.querySelector(selector));
  return result?.snapshot ?? null;
};
`,
);

try {
  await build({
    entryPoints: [entrySource],
    outfile: browserBundle,
    bundle: true,
    format: "iife",
    platform: "browser",
    target: "chrome120",
    logLevel: "silent",
  });

  const browser = await chromium.launch({
    executablePath: chromeExecutablePath,
    headless: true,
  });
  const page = await browser.newPage();
  try {
    await page.setContent(`
      <form id="login-form">
        <input id="login-id" name="username" type="text" value="child@example.com" />
        <input id="login-password" name="password" type="password" value="secret" />
      </form>
      <input id="email-field" type="email" value="child@example.com" />
      <input id="search-field" type="search" value="오늘 기분" />
      <textarea id="post-body" name="post_body">대단하네</textarea>
      <div id="blog-editor" contenteditable="true" aria-label="글 내용 입력">대단하네</div>
    `);
    await page.addScriptTag({ path: browserBundle });

    const snapshots = await page.evaluate(() => ({
      loginId: globalThis.readSnapshotForSelector("#login-id"),
      loginPassword: globalThis.readSnapshotForSelector("#login-password"),
      email: globalThis.readSnapshotForSelector("#email-field"),
      search: globalThis.readSnapshotForSelector("#search-field"),
      postBody: globalThis.readSnapshotForSelector("#post-body"),
      blogEditor: globalThis.readSnapshotForSelector("#blog-editor"),
    }));

    assert.equal(snapshots.loginId, null);
    assert.equal(snapshots.loginPassword, null);
    assert.equal(snapshots.email, null);
    assert.equal(snapshots.search?.text, "오늘 기분");
    assert.equal(snapshots.postBody?.text, "대단하네");
    assert.equal(snapshots.blogEditor, null);

    console.log(
      JSON.stringify(
        {
          ok: true,
          cases: ["sensitive-inputs", "ordinary-input-surfaces"],
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close();
  }
} finally {
  await rm(tempDir, { force: true, recursive: true });
}
