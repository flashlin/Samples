import { PcmTap } from "../shared/audio/pcm-tap";
import { isOffscreenCommand } from "../shared/message-guards";
import type { OffscreenEvent } from "../shared/messages";
import { audioMessage, TAB_STREAM_PORT_PREFIX } from "../shared/stream-protocol";

const MILLISECONDS_PER_SECOND = 1000;

interface TabCapture {
  readonly stream: MediaStream;
  readonly playback: AudioContext;
  readonly tap: PcmTap;
  readonly port: chrome.runtime.Port;
}

const captures = new Map<number, TabCapture>();

function emit(event: OffscreenEvent): void {
  void chrome.runtime.sendMessage(event).catch(() => undefined);
}

async function openTabStream(streamId: string): Promise<MediaStream> {
  const constraints = { audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } }, video: false };
  return navigator.mediaDevices.getUserMedia(constraints as MediaStreamConstraints);
}

async function startCapture(tabId: number, streamId: string): Promise<void> {
  const stream = await openTabStream(streamId);
  const playback = new AudioContext();
  playback.createMediaStreamSource(stream).connect(playback.destination);
  const port = chrome.runtime.connect({ name: `${TAB_STREAM_PORT_PREFIX}${tabId}` });
  const tap = new PcmTap(stream, {
    clock: () => Date.now() / MILLISECONDS_PER_SECOND,
    onFrame: (frame) => port.postMessage(audioMessage(frame)),
    onDiscontinuity: () => port.postMessage({ type: "stream-break" }),
  });
  captures.set(tabId, { stream, playback, tap, port });
  stream.getAudioTracks()[0]?.addEventListener("ended", () => stopCapture(tabId, "Tab audio capture ended"));
  tap.setActive(true);
}

function stopCapture(tabId: number, message: string | null): void {
  const capture = captures.get(tabId);
  if (!capture) {
    return;
  }
  captures.delete(tabId);
  capture.tap.stop();
  capture.port.disconnect();
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
    const { tabId, streamId } = message;
    startCapture(tabId, streamId).catch((error: unknown) => {
      emit({ type: "offscreen-ended", tabId, message: `Tab audio capture failed: ${error instanceof Error ? error.message : String(error)}` });
    });
  } else {
    stopCapture(message.tabId, null);
  }
});
