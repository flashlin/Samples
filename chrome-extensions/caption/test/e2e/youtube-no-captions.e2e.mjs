import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createWorkDir, launchExtensionWithServer, waitFor } from "./harness.mjs";

const VIDEO_URL = process.env.CAPTION_E2E_YOUTUBE_URL ?? "https://www.youtube.com/watch?v=mAbdM_F0sQI";
const FALLBACK_WAIT_MILLISECONDS = 90_000;

function savedMeta(dataDir) {
  const subtitles = path.join(dataDir, "subtitles");
  const metaFile = readdirSync(subtitles, { withFileTypes: true }).find((entry) => entry.name.endsWith(".meta.json"));
  if (!metaFile) return null;
  const meta = JSON.parse(readFileSync(path.join(subtitles, metaFile.name), "utf8"));
  const srt = readFileSync(path.join(subtitles, metaFile.name.replace(".meta.json", ".srt")), "utf8");
  return { meta, srt };
}

async function main() {
  const workDir = createWorkDir();
  const { context, dataDir, server, close } = await launchExtensionWithServer(workDir);
  try {
    const page = await context.newPage();
    await page.goto(VIDEO_URL, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("video", { timeout: 30_000 });
    await page.evaluate(() => {
      const video = document.querySelector("video");
      video.muted = false;
      return video.play();
    });
    const pressed = await waitFor(() => page.getAttribute(".ytp-subtitles-button", "aria-pressed"), 15_000, "subtitles button");
    await waitFor(() => server.transcribeCount() > 0, FALLBACK_WAIT_MILLISECONDS, "fallback to speech recognition");
    const saved = await waitFor(
      () => {
        try {
          const current = savedMeta(dataDir);
          return current && current.srt.trim().length > 0 ? current : null;
        } catch {
          return null;
        }
      },
      FALLBACK_WAIT_MILLISECONDS,
      "non-empty SRT saved from speech recognition",
    );
    await page.screenshot({ path: path.join(workDir, "youtube-fallback.png") });
    assert.equal(saved.meta.source, "asr");
    assert.equal(await page.getAttribute(".ytp-subtitles-button", "aria-pressed"), "false", "the CC button must be left as it was");
    console.log(JSON.stringify({ ok: true, workDir, ccButtonBefore: pressed, transcribeCalls: server.transcribeCount(), meta: saved.meta, srtHead: saved.srt.slice(0, 300) }, null, 1));
  } finally {
    await close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
