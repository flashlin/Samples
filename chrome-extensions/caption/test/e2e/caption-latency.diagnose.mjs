import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { createWorkDir, launchExtensionWithServer, readClosedShadowText } from "./harness.mjs";

const VIDEO_URL = process.env.CAPTION_E2E_YOUTUBE_URL ?? "https://www.youtube.com/watch?v=mAbdM_F0sQI";
const PLAY_SECONDS = Number(process.env.CAPTION_E2E_PLAY_SECONDS ?? 34);
const PROBE_CHARS = 3;

function parseSrt(text) {
  return text
    .trim()
    .split(/\n\s*\n/)
    .map((block) => block.split("\n"))
    .map(([, timing, ...lines]) => {
      const [start, end] = timing.split(" --> ").map((value) => {
        const [h, m, rest] = value.split(":");
        return Number(h) * 3600 + Number(m) * 60 + Number(rest.replace(",", "."));
      });
      return { start, end, text: lines.join("\n") };
    });
}

async function main() {
  const workDir = createWorkDir();
  const { context, dataDir, close } = await launchExtensionWithServer(workDir);
  try {
    const page = await context.newPage();
    await page.goto(VIDEO_URL, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("video", { timeout: 30_000 });
    await page.evaluate(() => document.querySelector("video").play());
    const polls = [];
    const startedAt = Date.now();
    while (Date.now() - startedAt < PLAY_SECONDS * 1000) {
      const [t, text] = await Promise.all([page.evaluate(() => document.querySelector("video").currentTime), readClosedShadowText(page, "text")]);
      polls.push({ t, text: text ?? "" });
      await page.waitForTimeout(300);
    }
    await page.evaluate(() => document.querySelector("video").pause());
    await page.waitForTimeout(12_000);
    const srtFile = readdirSync(path.join(dataDir, "subtitles")).find((name) => name.endsWith(".srt"));
    const cues = parseSrt(readFileSync(path.join(dataDir, "subtitles", srtFile), "utf8"));
    const rows = cues.map((cue) => {
      const probe = cue.text.replace(/[，。？！、\s]/g, "").slice(0, PROBE_CHARS);
      const shown = polls.find((poll) => poll.t >= cue.start - 1 && poll.text.replace(/[，。？！、\s]/g, "").includes(probe));
      return { spoken: `${cue.start.toFixed(1)}-${cue.end.toFixed(1)}`, text: cue.text.slice(0, 18), firstShownAt: shown ? shown.t.toFixed(1) : "never", lagFromStart: shown ? (shown.t - cue.start).toFixed(1) : null };
    });
    writeFileSync(path.join(workDir, "polls.json"), JSON.stringify(polls, null, 1));
    console.log(JSON.stringify({ workDir, rows }, null, 1));
  } finally {
    await close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
