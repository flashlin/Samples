import { CONFIG_STORAGE_KEY, LOOPBACK_HOSTS, parseCaptionConfig, parseCaptionConfigOrDefault, type CaptionConfig } from "../shared/config";
import { askBackgroundFromPage } from "../shared/extension-page-client";

function field<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`Missing options element #${id}`);
  }
  return found as T;
}

const endpoint = field<HTMLInputElement>("endpoint");
const token = field<HTMLInputElement>("token");
const siteMode = field<HTMLSelectElement>("site-mode");
const allowlist = field<HTMLTextAreaElement>("allowlist");
const denylist = field<HTMLTextAreaElement>("denylist");
const fontSize = field<HTMLInputElement>("font-size");
const heightLines = field<HTMLInputElement>("height-lines");
const useYoutubeCaptions = field<HTMLInputElement>("use-youtube-captions");
const autoStart = field<HTMLInputElement>("auto-start");
const statusText = field<HTMLElement>("status");

function lines(textarea: HTMLTextAreaElement): string[] {
  return textarea.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

async function loadStoredRaw(): Promise<Record<string, unknown>> {
  const stored = await chrome.storage.local.get(CONFIG_STORAGE_KEY);
  return (stored[CONFIG_STORAGE_KEY] as Record<string, unknown> | undefined) ?? {};
}

function render(config: CaptionConfig): void {
  endpoint.value = config.server.endpoint;
  token.value = config.server.token;
  siteMode.value = config.sites.mode;
  allowlist.value = config.sites.allowlist.join("\n");
  denylist.value = config.sites.denylist.join("\n");
  fontSize.value = String(config.overlay.fontSizePx);
  heightLines.value = String(config.overlay.heightLines);
  useYoutubeCaptions.checked = config.features.useYoutubeCaptions;
  autoStart.checked = config.features.autoStart;
}

function collect(current: CaptionConfig): unknown {
  return {
    ...current,
    server: { ...current.server, endpoint: endpoint.value.trim(), token: token.value.trim() },
    sites: { mode: siteMode.value, allowlist: lines(allowlist), denylist: lines(denylist) },
    overlay: { ...current.overlay, fontSizePx: Number(fontSize.value), heightLines: Number(heightLines.value) },
    features: { ...current.features, useYoutubeCaptions: useYoutubeCaptions.checked, autoStart: autoStart.checked },
  };
}

async function ensureEndpointPermission(endpointUrl: string): Promise<boolean> {
  const url = new URL(endpointUrl);
  if (LOOPBACK_HOSTS.has(url.hostname)) {
    return true;
  }
  return chrome.permissions.request({ origins: [`${url.origin}/*`] });
}

async function save(): Promise<void> {
  const { config: base } = parseCaptionConfigOrDefault(await loadStoredRaw());
  const next = parseCaptionConfig(collect(base));
  if (!next.ok) {
    statusText.textContent = next.errors.join("\n");
    return;
  }
  if (!(await ensureEndpointPermission(next.config.server.endpoint))) {
    statusText.textContent = "Permission to reach the caption server was not granted";
    return;
  }
  await chrome.storage.local.set({ [CONFIG_STORAGE_KEY]: next.config });
  statusText.textContent = "Saved";
}

async function testConnection(): Promise<void> {
  try {
    const health = await askBackgroundFromPage<{ engine: Record<string, string> }>({ type: "check-server" });
    statusText.textContent = `Connected · ASR ${health.engine.asr} · MT ${health.engine.translate}`;
  } catch (error) {
    statusText.textContent = error instanceof Error ? error.message : String(error);
  }
}

async function main(): Promise<void> {
  const { config, errors } = parseCaptionConfigOrDefault(await loadStoredRaw());
  render(config);
  statusText.textContent = errors.join("\n");
  field<HTMLButtonElement>("save").addEventListener("click", () => void save());
  field<HTMLButtonElement>("test").addEventListener("click", () => void testConnection());
}

void main();
