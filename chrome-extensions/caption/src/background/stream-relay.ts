import { base64ToBytes } from "../shared/base64";
import { isStreamClientMessage, type StreamClientMessage, type StreamServerMessage } from "../shared/stream-protocol";

const STREAM_PATH = "/v1/captions/stream";
const FRAME_HEADER_BYTES = 8;
const MAX_QUEUED_MESSAGES = 200;

export interface RelayConnection {
  readonly endpoint: string;
  readonly token: string;
}

function streamUrl(endpoint: string): string {
  const url = new URL(STREAM_PATH, endpoint);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function encodeFrame(start: number, pcmBase64: string): ArrayBuffer {
  const pcm = base64ToBytes(pcmBase64);
  const frame = new Uint8Array(FRAME_HEADER_BYTES + pcm.length);
  new DataView(frame.buffer).setFloat64(0, start, true);
  frame.set(pcm, FRAME_HEADER_BYTES);
  return frame.buffer;
}

export class StreamRelay {
  private readonly socket: WebSocket;
  private ready = false;
  private closed = false;
  private readonly queued: (ArrayBuffer | string)[] = [];

  constructor(
    connection: RelayConnection,
    private readonly deliver: (message: StreamServerMessage) => void,
  ) {
    this.socket = new WebSocket(streamUrl(connection.endpoint));
    this.socket.binaryType = "arraybuffer";
    this.socket.addEventListener("open", () => this.socket.send(JSON.stringify({ type: "start", token: connection.token })));
    this.socket.addEventListener("message", (event) => this.receive(event.data));
    this.socket.addEventListener("close", (event) => this.finish(event.code === 1000 ? null : `Caption server stream closed (${event.code})`));
    this.socket.addEventListener("error", () => this.finish("Caption server stream failed, is caption-server running?"));
  }

  send(message: StreamClientMessage): void {
    const payload = message.type === "stream-audio" ? encodeFrame(message.start, message.pcm) : JSON.stringify({ type: "break" });
    if (this.ready) {
      this.socket.send(payload);
      return;
    }
    this.queued.push(payload);
    if (this.queued.length > MAX_QUEUED_MESSAGES) {
      this.queued.shift();
    }
  }

  close(): void {
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "stop" }));
    } else {
      this.socket.close();
    }
  }

  private receive(data: unknown): void {
    if (typeof data !== "string") {
      return;
    }
    const message = JSON.parse(data) as StreamServerMessage;
    if (message.type === "ready") {
      this.ready = true;
      for (const payload of this.queued.splice(0)) {
        this.socket.send(payload);
      }
    }
    this.deliver(message);
  }

  private finish(message: string | null): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.deliver({ type: "closed", message });
  }
}

export function relayPort(port: chrome.runtime.Port, connection: Promise<RelayConnection>, deliver: (message: StreamServerMessage) => void): void {
  const pending: StreamClientMessage[] = [];
  let relay: StreamRelay | null = null;
  let disconnected = false;
  port.onMessage.addListener((message: unknown) => {
    if (!isStreamClientMessage(message)) {
      return;
    }
    if (relay) {
      relay.send(message);
    } else if (pending.length < MAX_QUEUED_MESSAGES) {
      pending.push(message);
    }
  });
  port.onDisconnect.addListener(() => {
    disconnected = true;
    relay?.close();
  });
  connection
    .then((resolved) => {
      if (disconnected) {
        return;
      }
      relay = new StreamRelay(resolved, deliver);
      for (const message of pending.splice(0)) {
        relay.send(message);
      }
    })
    .catch((error: unknown) => deliver({ type: "closed", message: error instanceof Error ? error.message : String(error) }));
}
