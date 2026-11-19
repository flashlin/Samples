import type { RecognitionSession } from "../shared/progress/recognition-session";
import { askBackground } from "./background-client";

const MILLISECONDS_PER_SECOND = 1000;
const CONFIRMATION_REQUIRED = "confirmation-required";

export class SubtitleSaver {
  private timer: ReturnType<typeof setInterval> | null = null;
  private inFlight: Promise<void> | null = null;
  private stopped = false;

  constructor(
    private readonly session: RecognitionSession,
    private readonly urlKey: string,
    private revision: number,
    private readonly intervalSeconds: number,
    private readonly onSaved: () => void,
  ) {}

  start(): void {
    this.timer ??= setInterval(() => void this.flush(), this.intervalSeconds * MILLISECONDS_PER_SECOND);
  }

  async stop(): Promise<void> {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    await this.flush();
    this.stopped = true;
  }

  flush(): Promise<void> {
    if (this.stopped) {
      return Promise.resolve();
    }
    this.inFlight ??= this.saveSnapshot().finally(() => {
      this.inFlight = null;
    });
    return this.inFlight;
  }

  private async saveSnapshot(): Promise<void> {
    if (this.session.awaitingOverwriteDecision) {
      return;
    }
    const snapshot = this.session.takeSnapshotIfWorthSaving();
    if (!snapshot) {
      return;
    }
    try {
      const result = await askBackground({
        type: "save-subtitle",
        urlKey: this.urlKey,
        meta: snapshot.meta,
        cues: [...snapshot.cues],
        placement: snapshot.placement,
        expectedRevision: this.revision,
        confirmOverwrite: snapshot.confirmOverwrite,
      });
      this.revision = result.revision;
      if (result.written) {
        this.session.acknowledgeSaved(snapshot);
        this.onSaved();
      } else if (result.reason === CONFIRMATION_REQUIRED) {
        this.session.markExistingReplacedElsewhere();
      } else {
        this.session.markDirty();
      }
    } catch (error) {
      this.session.markDirty();
      console.warn("Caption subtitle save failed", error);
    }
  }
}
