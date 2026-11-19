import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import assert from "node:assert/strict";
import { createWorkDir, findButtonInClosedShadow, launchExtensionWithServer, waitFor } from "./harness.mjs";

const workDir = createWorkDir();
const PAGE_PORT = 8811;
const SAMPLE_RATE = 16000;

const SPEECH = [
  { voice: "Meijia", text: "接下來我們要介紹 transformer 的 encoder 跟 decoder。" },
  { voice: "Samantha", text: "Today we are going to talk about attention." },
  { voice: "Meijia", text: "這個模型的 performance 越來越好。" },
];

function pcmFromSay(voice, text) {
  const file = path.join(workDir, `say-${voice}-${Math.random().toString(36).slice(2)}.wav`);
  execFileSync("say", ["-v", voice, "--file-format=WAVE", `--data-format=LEI16@${SAMPLE_RATE}`, "-o", file, text]);
  const buffer = readFileSync(file);
  const dataOffset = buffer.indexOf("data") + 8;
  return buffer.subarray(dataOffset);
}

function wavFromPcm(pcm) {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVEfmt ", 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(SAMPLE_RATE, 24);
  header.writeUInt32LE(SAMPLE_RATE * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

function buildFixtureWav() {
  const silence = Buffer.alloc(SAMPLE_RATE * 2);
  const parts = [silence];
  for (const { voice, text } of SPEECH) {
    parts.push(pcmFromSay(voice, text), silence);
  }
  return wavFromPcm(Buffer.concat(parts));
}

function servePage(wav) {
  const html = `<!doctype html><meta charset="utf-8"><title>caption e2e</title>
<body style="margin:0;background:#222"><video id="v" src="/speech.wav" width="640" height="360" controls></video></body>`;
  const server = http.createServer((req, res) => {
    if (req.url?.startsWith("/speech.wav")) {
      res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": wav.length, "Accept-Ranges": "none" });
      res.end(wav);
      return;
    }
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(html);
  });
  return new Promise((resolve) => server.listen(PAGE_PORT, "127.0.0.1", () => resolve(server)));
}

async function playVideo(page) {
  await page.goto(`http://127.0.0.1:${PAGE_PORT}/`);
  await page.evaluate(() => document.getElementById("v").play());
}

function readSavedSubtitle(dataDir) {
  try {
    const files = execFileSync("ls", [path.join(dataDir, "subtitles")]).toString().split("\n");
    const meta = files.find((name) => name.endsWith(".meta.json"));
    if (!meta) return null;
    const parsed = JSON.parse(readFileSync(path.join(dataDir, "subtitles", meta), "utf8"));
    return { meta: parsed, srt: readFileSync(path.join(dataDir, "subtitles", meta.replace(".meta.json", ".srt")), "utf8") };
  } catch {
    return null;
  }
}

const currentTime = (page) => page.evaluate(() => document.getElementById("v").currentTime);

async function recordPartialThenLeave(context, dataDir) {
  const page = await context.newPage();
  await playVideo(page);
  await waitFor(() => page.locator("[data-caption-extension]").count(), 15000, "caption overlay");
  await waitFor(async () => (await currentTime(page)) >= 0.5, 10000, "playback start");
  await page.screenshot({ path: path.join(workDir, "silence-0.5s.png") });
  await waitFor(async () => (await currentTime(page)) >= 3, 10000, "3 seconds of playback");
  await page.screenshot({ path: path.join(workDir, "interim-3s.png") });
  const partial = await waitFor(() => readSavedSubtitle(dataDir), 20000, "first partial save");
  await page.close();
  return partial;
}

async function replaceAfterConfirmation(context, dataDir, first) {
  const page = await context.newPage();
  await playVideo(page);
  const duration = await page.evaluate(() => document.getElementById("v").duration);
  await waitFor(async () => (await currentTime(page)) >= 11, 20000, "11 seconds of playback");
  assert.equal(await findButtonInClosedShadow(page, "Replace"), null, "the overwrite question must wait until recording stops");
  await page.screenshot({ path: path.join(workDir, "recognizing.png") });
  const replace = await waitFor(() => findButtonInClosedShadow(page, "Replace"), duration * 1000 + 15000, "overwrite question after the video ended");
  await page.screenshot({ path: path.join(workDir, "overwrite-question.png") });
  assert.equal(readSavedSubtitle(dataDir).meta.sessionId, first.meta.sessionId, "nothing may be overwritten before the user answers");
  await page.mouse.click(replace.x, replace.y);
  const replaced = await waitFor(
    () => {
      const saved = readSavedSubtitle(dataDir);
      return saved && saved.meta.sessionId !== first.meta.sessionId && saved.meta.status === "complete" ? saved : null;
    },
    20000,
    "complete subtitle written by the confirmed session",
  );
  await page.close();
  return { replaced, duration };
}

async function main() {
  const wav = buildFixtureWav();
  const pageServer = await servePage(wav);
  const { context, worker, extensionId, dataDir, server, close } = await launchExtensionWithServer(workDir);
  try {
    const first = await recordPartialThenLeave(context, dataDir);
    assert.equal(first.meta.status, "partial");
    const { replaced: subtitleFile, duration } = await replaceAfterConfirmation(context, dataDir, first);
    const streamsBeforeReplay = server.streamCount();
    assert.ok(streamsBeforeReplay >= 2, `both recording visits must open a recognition stream, saw ${streamsBeforeReplay}`);
    writeFileSync(path.join(workDir, "result.srt"), subtitleFile.srt);
    assert.equal(subtitleFile.meta.canonicalUrl, `http://127.0.0.1:${PAGE_PORT}/`);
    assert.ok(subtitleFile.meta.coveredSeconds >= duration * 0.95, `coverage ${subtitleFile.meta.coveredSeconds} of ${duration}`);
    assert.match(subtitleFile.srt, /transformer/i);
    assert.match(subtitleFile.srt, /attention/i);
    assert.match(subtitleFile.srt, /[\u4e00-\u9fff]/);

    const second = await context.newPage();
    await playVideo(second);
    await waitFor(() => second.locator("[data-caption-extension]").count(), 15000, "overlay on playback visit");
    await second.waitForTimeout(3000);
    await second.screenshot({ path: path.join(workDir, "playback.png") });
    assert.equal(server.streamCount(), streamsBeforeReplay, "a complete subtitle must be replayed without ASR calls");

    await second.mouse.move(320, 290);
    await second.mouse.down();
    await second.mouse.move(320, 200, { steps: 5 });
    await second.mouse.move(320, 150, { steps: 5 });
    await second.mouse.up();
    await second.screenshot({ path: path.join(workDir, "dragged.png") });
    const savedOffset = await waitFor(
      () => worker.evaluate(() => chrome.storage.local.get("captionConfig").then((stored) => stored.captionConfig?.overlay ?? null)),
      5000,
      "saved overlay offset",
    );
    assert.ok(savedOffset.offsetY < -0.3, `overlay offset should move up, got ${JSON.stringify(savedOffset)}`);
    assert.ok(Math.abs(savedOffset.offsetX) < 0.05);

    const options = await context.newPage();
    await options.goto(`chrome-extension://${extensionId}/options.html`);
    await options.click("#test");
    await waitFor(async () => (await options.textContent("#status"))?.startsWith("Connected"), 10000, "options page connection test");

    console.log(JSON.stringify({ ok: true, workDir, duration, firstSession: first.meta.sessionId, replacedBy: subtitleFile.meta.sessionId, coveredSeconds: subtitleFile.meta.coveredSeconds, srt: subtitleFile.srt }, null, 1));
  } finally {
    await close();
    pageServer.close();
  }
}

main().catch((error) => {
  console.error(error);
  console.error(`Artifacts in ${workDir}`);
  process.exit(1);
});
