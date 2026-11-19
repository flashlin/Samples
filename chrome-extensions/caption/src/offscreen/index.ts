import { AudioTap, type AudioChunk } from "../shared/audio/audio-tap";
import type { ChunkLimits } from "../shared/audio/chunk-cut-policy";
import { blobToBase64 } from "../shared/base64";
import { isOffscreenCommand } from "../shared/message-guards";
import type { OffscreenEvent } from "../shared/messages";

const MILLISECONDS_PER_SECOND = 1000;

interface TabCapture {
  readonly stream: MediaStream;
  readonly playback: AudioContext;
  readonly tap: AudioTap;
}

const captures = new Map<number, TabCapture>();

function emit(event: OffscreenEvent): void {
  void chrome.runtime.sendMessage(event).catch(() => undefined);
}

async function forwardChunk(tabId: number, chunk: AudioChunk): Promise<void> {
  if (chunk.silent) {
    emit({ type: "offscreen-silence", tabId, start: chunk.start, end: chunk.end });
    return;
  }
  emit({ type: "offscreen-chunk", tabId, start: chunk.start, end: chunk.end, audioBase64: await blobToBase64(chunk.blob), mimeType: chunk.blob.type });
}

async function openTabStream(streamId: string): Promise<MediaStream> {
  const constraints = { audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } }, video: false };
  return navigator.mediaDevices.getUserMedia(constraints as MediaStreamConstraints);
}

async function startCapture(tabId: number, streamId: string, limits: ChunkLimits): Promise<void> {
  const stream = await openTabStream(streamId);
  const playback = new AudioContext();
  playback.createMediaStreamSource(stream).connect(playback.destination);
  const tap = new AudioTap(stream, { limits, clock: () => Date.now() / MILLISECONDS_PER_SECOND, onChunk: (chunk) => void forwardChunk(tabId, chunk) });
  captures.set(tabId, { stream, playback, tap });
  stream.getAudioTracks()[0]?.addEventListener("ended", () => stopCapture(tabId, "Tab audio capture ended"));
  tap.resume();
}

function stopCapture(tabId: number, message: string | null): void {
  const capture = captures.get(tabId);
  if (!capture) {
    return;
  }
  captures.delete(tabId);
  capture.tap.stop();
  for (const track of capture.stream.getTracks()) {
    track.stop();
  }
  void capture.playback.close();
  emit({ type: "offscreen-ended", tabId, message });
}

chrome.runtime.onMessage.addListener((message) => {
  if (!isOffscreenCommand(message)) {
    return;
  }
  if (message.type === "offscreen-start") {
    const { tabId, streamId, limits } = message;
    startCapture(tabId, streamId, limits).catch((error: unknown) => {
      emit({ type: "offscreen-ended", tabId, message: `Tab audio capture failed: ${error instanceof Error ? error.message : String(error)}` });
    });
  } else {
    stopCapture(message.tabId, null);
  }
});
