import path from "node:path";
import { createWorkDir, launchExtensionWithServer, waitFor } from "./harness.mjs";

const VIDEO_URL = process.env.CAPTION_E2E_YOUTUBE_URL ?? "https://www.youtube.com/watch?v=mAbdM_F0sQI";
const PLAY_SECONDS = Number(process.env.CAPTION_E2E_PLAY_SECONDS ?? 60);
const CONFIG_OVERRIDE = JSON.parse(process.env.CAPTION_E2E_CONFIG ?? "{}");

async function main() {
  const workDir = createWorkDir();
  const { context, worker, close } = await launchExtensionWithServer(workDir);
  try {
    if (Object.keys(CONFIG_OVERRIDE).length > 0) {
      await worker.evaluate(async (override) => {
        const stored = (await chrome.storage.local.get("captionConfig")).captionConfig ?? {};
        await chrome.storage.local.set({ captionConfig: { ...stored, ...override } });
      }, CONFIG_OVERRIDE);
    }
    const page = await context.newPage();
    const consoleLines = [];
    page.on("console", (message) => {
      if (["error", "warning"].includes(message.type())) consoleLines.push(`${message.type()}: ${message.text().slice(0, 200)}`);
    });
    await page.goto(VIDEO_URL, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("video", { timeout: 30_000 });
    await page.evaluate(() => document.querySelector("video").play());
    const samples = [];
    const startedAt = Date.now();
    while (Date.now() - startedAt < PLAY_SECONDS * 1000) {
      await page.waitForTimeout(5000);
      samples.push(
        await page.evaluate(() => {
          const video = document.querySelector("video");
          return {
            t: Math.round(video.currentTime * 10) / 10,
            paused: video.paused,
            readyState: video.readyState,
            error: video.error ? `${video.error.code}:${video.error.message}` : null,
            playerError: document.querySelector(".ytp-error")?.textContent?.trim().slice(0, 60) ?? null,
            src: video.src.slice(0, 30),
          };
        }),
      );
    }
    await page.screenshot({ path: path.join(workDir, "long-play.png") });
    console.log(JSON.stringify({ workDir, samples, console: consoleLines.slice(-15) }, null, 1));
  } finally {
    await close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
