import { CONFIG_STORAGE_KEY, parseCaptionConfigOrDefault, type CaptionConfig } from "../shared/config";
import type { ContentSettings } from "../shared/messages";

export async function loadConfig(): Promise<CaptionConfig> {
  const stored = await chrome.storage.local.get(CONFIG_STORAGE_KEY);
  const { config, errors } = parseCaptionConfigOrDefault(stored[CONFIG_STORAGE_KEY]);
  if (errors.length > 0) {
    console.warn("Stored caption config is invalid, falling back to defaults", errors);
  }
  return config;
}

export function toContentSettings(config: CaptionConfig): ContentSettings {
  return {
    chunk: { minSeconds: config.chunk.minSeconds, maxSeconds: config.chunk.maxSeconds, silenceSeconds: config.chunk.silenceSeconds },
    interimSeconds: config.chunk.interimSeconds,
    flushIntervalSeconds: config.flush.intervalSeconds,
    completionRatio: config.completion.ratio,
    queue: config.queue,
    overlay: config.overlay,
    features: {
      autoStart: config.features.autoStart,
      useYoutubeCaptions: config.features.useYoutubeCaptions,
      recognizeOnlyAtNormalRate: config.features.recognizeOnlyAtNormalRate,
    },
  };
}

export async function saveOverlayOffset(x: number, y: number): Promise<void> {
  const config = await loadConfig();
  await chrome.storage.local.set({ [CONFIG_STORAGE_KEY]: { ...config, overlay: { ...config.overlay, offsetX: x, offsetY: y } } });
}
