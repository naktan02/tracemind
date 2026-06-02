import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(__dirname, "..");
const extensionPath = path.join(appRoot, "dist");
const fixturePath = path.join(appRoot, "collector-fixture.html");
const chromeExecutablePath =
  process.env.PLAYWRIGHT_CHROME_EXECUTABLE ?? "/usr/bin/google-chrome";

const server = await startFixtureServer();
const userDataDir = await mkdtemp(path.join(tmpdir(), "tracemind-extension-"));
const context = await chromium.launchPersistentContext(userDataDir, {
  executablePath: chromeExecutablePath,
  headless: false,
  args: [
    `--disable-extensions-except=${extensionPath}`,
    `--load-extension=${extensionPath}`,
    "--no-first-run",
    "--no-default-browser-check",
  ],
});

try {
  const extensionId = await resolveExtensionId(context);
  const debugPage = await context.newPage();
  await debugPage.goto(`chrome-extension://${extensionId}/collector-debug.html`);
  await enableDebug(debugPage);

  const fixturePage = await context.newPage();
  await fixturePage.goto(server.url);
  await fixturePage.getByLabel("Textarea surface").fill("playwright fixture smoke");
  await debugPage.waitForFunction(
    () => {
      const element = document.querySelector("#last-segment");
      return element?.textContent?.includes("playwright fixture smoke");
    },
    null,
    { timeout: 8000 },
  );

  const rawSegment = await debugPage.locator("#last-segment").textContent();
  const parsedSegment = JSON.parse(rawSegment ?? "{}");
  if (parsedSegment.final_text !== "playwright fixture smoke") {
    throw new Error(
      `Unexpected final_text: ${JSON.stringify(parsedSegment.final_text)}`,
    );
  }

  console.log(
    JSON.stringify(
      {
        ok: true,
        extensionId,
        fixtureUrl: server.url,
        segmentId: parsedSegment.segment_id,
        surfaceType: parsedSegment.surface_type,
      },
      null,
      2,
    ),
  );
} finally {
  await context.close();
  await new Promise((resolve) => server.instance.close(resolve));
}

async function resolveExtensionId(context) {
  let [serviceWorker] = context.serviceWorkers();
  if (serviceWorker === undefined) {
    serviceWorker = await context.waitForEvent("serviceworker", {
      timeout: 8000,
    });
  }
  const url = serviceWorker.url();
  const match = url.match(/^chrome-extension:\/\/([^/]+)\//);
  if (match === null) {
    throw new Error(`Unable to resolve extension id from ${url}`);
  }
  return match[1];
}

async function enableDebug(page) {
  const button = page.locator("#toggle-debug");
  await button.waitFor({ timeout: 5000 });
  const label = await button.textContent();
  if (label?.includes("켜기")) {
    await button.click();
  }
}

function startFixtureServer() {
  return new Promise((resolve, reject) => {
    const instance = createServer((request, response) => {
      if (request.url !== "/" && request.url !== "/collector-fixture.html") {
        response.writeHead(404);
        response.end("not found");
        return;
      }
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end(
        // HTML은 소스 fixture를 그대로 쓰고, 서버는 origin 부여만 담당한다.
        readFileSync(fixturePath),
      );
    });
    instance.once("error", reject);
    instance.listen(0, "127.0.0.1", () => {
      const address = instance.address();
      if (address === null || typeof address === "string") {
        reject(new Error("Unable to bind fixture server"));
        return;
      }
      resolve({
        instance,
        url: `http://127.0.0.1:${address.port}/collector-fixture.html`,
      });
    });
  });
}
