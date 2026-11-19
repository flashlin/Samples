import { chromium } from "playwright-core";
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

export const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
export const SERVER_PORT = 8766;

export function createWorkDir() {
  return mkdtempSync(path.join(process.env.CAPTION_E2E_TMP ?? tmpdir(), "caption-e2e-"));
}

function startCaptionServer(extensionId, dataDir) {
  const log = [];
  const child = spawn("uv", ["run", "caption-server"], {
    cwd: path.join(root, "server"),
    env: { ...process.env, HF_HUB_OFFLINE: "1", CAPTION_DATA_DIR: dataDir, CAPTION_PORT: String(SERVER_PORT), CAPTION_ALLOWED_ORIGINS: `chrome-extension://${extensionId}` },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const ready = new Promise((resolve, reject) => {
    const onData = (chunk) => {
      const text = chunk.toString();
      log.push(text);
      if (text.includes("Application startup complete")) resolve();
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("exit", (code) => reject(new Error(`caption-server exited with ${code}\n${log.join("")}`)));
  });
  return { child, ready, streamCount: () => (log.join("").match(/\/v1\/captions\/stream" \[accepted\]/g) ?? []).length };
}

export async function launchExtensionWithServer(workDir) {
  const extensionDir = path.join(root, "dist");
  const context = await chromium.launchPersistentContext(path.join(workDir, "profile"), {
    channel: "chromium",
    headless: true,
    args: [`--disable-extensions-except=${extensionDir}`, `--load-extension=${extensionDir}`, "--autoplay-policy=no-user-gesture-required"],
  });
  const worker = context.serviceWorkers()[0] ?? (await context.waitForEvent("serviceworker"));
  const extensionId = new URL(worker.url()).host;
  const dataDir = path.join(workDir, "server-data");
  const server = startCaptionServer(extensionId, dataDir);
  const close = async () => {
    await context.close();
    server.child.kill();
  };
  try {
    await server.ready;
  } catch (error) {
    await close();
    throw error;
  }
  const token = readFileSync(path.join(dataDir, "token"), "utf8").trim();
  const endpoint = `http://127.0.0.1:${SERVER_PORT}`;
  await worker.evaluate(({ value, url }) => chrome.storage.local.set({ captionConfig: { server: { token: value, endpoint: url } } }), { value: token, url: endpoint });
  return { context, worker, extensionId, dataDir, server, close };
}

export async function waitFor(check, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for ${label}`);
}

function findNodeId(node, predicate) {
  if (predicate(node)) return node.backendNodeId;
  for (const child of [...(node.children ?? []), ...(node.shadowRoots ?? [])]) {
    const found = findNodeId(child, predicate);
    if (found) return found;
  }
  return null;
}

export async function findButtonInClosedShadow(page, label) {
  const session = await page.context().newCDPSession(page);
  try {
    const { root } = await session.send("DOM.getDocument", { depth: -1, pierce: true });
    const isButtonWithLabel = (node) => node.nodeName === "BUTTON" && (node.children ?? []).some((child) => child.nodeType === 3 && child.nodeValue === label);
    const backendNodeId = findNodeId(root, isButtonWithLabel);
    if (!backendNodeId) return null;
    const { model } = await session.send("DOM.getBoxModel", { backendNodeId });
    const [x1, y1, , , x3, y3] = model.content;
    return { x: (x1 + x3) / 2, y: (y1 + y3) / 2 };
  } catch {
    return null;
  } finally {
    await session.detach();
  }
}

export async function readClosedShadowText(page, className) {
  const session = await page.context().newCDPSession(page);
  try {
    const { root } = await session.send("DOM.getDocument", { depth: -1, pierce: true });
    const hasClass = (node) => (node.attributes ?? []).some((value, index, all) => index % 2 === 1 && all[index - 1] === "class" && value.split(" ").includes(className));
    const collectText = (node) => (node.nodeType === 3 ? node.nodeValue : (node.children ?? []).map(collectText).join(""));
    const find = (node) => {
      if (hasClass(node)) return node;
      for (const child of [...(node.children ?? []), ...(node.shadowRoots ?? [])]) {
        const found = find(child);
        if (found) return found;
      }
      return null;
    };
    const node = find(root);
    return node ? collectText(node) : null;
  } finally {
    await session.detach();
  }
}
