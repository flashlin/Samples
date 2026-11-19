import type { Cue, SegmentSink } from "../shared/cue/cue";
import { IDLE_TAB_STATUS, type ContentSettings, type ResolvedPage, type TabBroadcast, type TabStatus } from "../shared/messages";
import { RecognitionSession } from "../shared/progress/recognition-session";
import { askBackground } from "./background-client";
import { captionTextFor, LIVE_CAPTION_HOLD_SECONDS } from "./caption-display";
import { CaptionOverlay } from "./caption-overlay";
import { findPlayingMedia, watchForPlayingMedia } from "./media-detector";
import { buildRunStatus } from "./run-status";
import { SubtitleSaver } from "./subtitle-saver";
import { TabCaptureReceiver } from "./tab-capture-receiver";
import { CrossOriginMediaError, StreamingRecognizer } from "./streaming-recognizer";
import { disableYoutubeCaptions, enableYoutubeCaptions, isYoutubeAdPlaying, isYoutubePage } from "./youtube";
import { YoutubeCaptionReader } from "./youtube-caption-reader";

const DISPLAY_REFRESH_MILLISECONDS = 200;
const STATUS_REFRESH_MILLISECONDS = 1000;
const PAUSE_BEFORE_ASKING_MILLISECONDS = 800;
const NO_MEDIA_MESSAGE = "No video or audio is playing on this page. Use Capture tab audio instead.";

type ResolvedForCaptions = Extract<ResolvedPage, { mode: "playback" | "recognize" }>;

interface ActiveRun {
  readonly resolved: ResolvedForCaptions;
  readonly overlay: CaptionOverlay;
  readonly timers: ReturnType<typeof setInterval>[];
  media: HTMLMediaElement | null;
  mode: TabStatus["mode"];
  session: RecognitionSession | null;
  saver: SubtitleSaver | null;
  sink: SegmentSink | null;
  overwriteQuestion: Promise<void> | null;
  stopSource: (() => void) | null;
  liveCue: () => Cue | null;
  tabCapture: TabCaptureReceiver | null;
  watchingMediaEnd: boolean;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class PageCaptionController {
  private run: ActiveRun | null = null;
  private resolving = false;
  private closed = false;
  private stopWatching: (() => void) | null = null;

  constructor(private readonly pageUrl: string) {}

  start(): void {
    this.stopWatching ??= watchForPlayingMedia((media) => void this.onMediaPlaying(media));
  }

  async dispose(): Promise<void> {
    this.stopWatching?.();
    this.stopWatching = null;
    await this.teardownRun();
  }

  async handleBroadcast(message: TabBroadcast): Promise<void> {
    switch (message.type) {
      case "tab-capture-started":
        return this.beginTabCapture();
      case "tab-capture-segment":
        this.run?.tabCapture?.receive(message.start, message.end, message.display);
        return;
      case "tab-capture-partial":
        this.run?.tabCapture?.receivePartial(message.start, message.end, message.text);
        return;
      case "tab-capture-stopped":
        return this.endTabCapture(message.message);
      case "subtitle-changed":
        return this.reload();
      case "recognition-start":
        return this.startRecognitionByUser();
      case "recognition-stop":
        return this.stopRecognitionByUser(message.discard);
      case "save-now":
        return this.saveNow();
      case "seek":
        this.seek(message.time);
        return;
    }
  }

  private async reload(): Promise<void> {
    await this.dispose();
    this.closed = false;
    this.start();
  }

  private async onMediaPlaying(media: HTMLMediaElement): Promise<void> {
    if (this.closed || this.resolving) {
      return;
    }
    if (this.run) {
      this.run.media ??= media;
      return;
    }
    this.resolving = true;
    try {
      const run = await this.openRun(media);
      if (run?.resolved.mode === "playback") {
        run.mode = "playback";
        this.reportRun(run);
      } else if (run) {
        await this.startAutomaticRecognition(run, media);
      }
    } catch (error) {
      this.report({ mode: "error", message: describe(error) });
    } finally {
      this.resolving = false;
    }
  }

  private async openRun(media: HTMLMediaElement | null): Promise<ActiveRun | null> {
    const resolved = await askBackground({ type: "resolve-page", pageUrl: this.pageUrl });
    if (resolved.mode === "disabled") {
      this.report({ mode: "disabled", message: resolved.reason });
      this.stopWatching?.();
      return null;
    }
    const overlay = new CaptionOverlay(resolved.settings.overlay, {
      onClose: () => void this.closeByUser(),
      onMoved: (offset) => void askBackground({ type: "save-overlay-offset", x: offset.x, y: offset.y }).catch(() => undefined),
    });
    overlay.attachTo(media);
    const run: ActiveRun = {
      resolved,
      overlay,
      media,
      mode: "idle",
      session: null,
      saver: null,
      sink: null,
      overwriteQuestion: null,
      stopSource: null,
      liveCue: () => null,
      tabCapture: null,
      watchingMediaEnd: false,
      timers: [
        setInterval(() => overlay.showCaption(this.displayText(run)), DISPLAY_REFRESH_MILLISECONDS),
        setInterval(() => this.reportRun(run), STATUS_REFRESH_MILLISECONDS),
      ],
    };
    this.run = run;
    return run;
  }

  private displayText(run: ActiveRun): string | null {
    return captionTextFor({
      liveCue: run.liveCue(),
      liveCues: run.session?.liveCues ?? null,
      liveCoverage: run.session?.liveCoverage ?? [],
      savedCues: run.resolved.cues,
      time: run.media?.currentTime ?? null,
      maxSentences: run.resolved.settings.overlay.maxSentences,
    });
  }

  private async startAutomaticRecognition(run: ActiveRun, media: HTMLMediaElement): Promise<void> {
    const settings = run.resolved.settings;
    const youtubeStarted = isYoutubePage() && settings.features.useYoutubeCaptions && (await this.tryYoutubeCaptions(run, media));
    if (!youtubeStarted && settings.features.autoStart) {
      this.startMediaAsr(run, media, settings);
    }
    this.reportRun(run);
  }

  private async startRecognitionByUser(): Promise<void> {
    if (window !== window.top) {
      return;
    }
    const media = this.run?.media ?? findPlayingMedia() ?? document.querySelector<HTMLMediaElement>("video, audio");
    if (!media) {
      this.run?.overlay.showMessage(NO_MEDIA_MESSAGE);
      this.report({ mode: "error", message: NO_MEDIA_MESSAGE });
      return;
    }
    this.closed = false;
    const run = this.run ?? (await this.openRun(media));
    if (!run || run.stopSource) {
      return;
    }
    run.media ??= media;
    run.overlay.attachTo(media);
    this.startMediaAsr(run, media, run.resolved.settings);
    this.reportRun(run);
  }

  private async stopRecognitionByUser(discard: boolean): Promise<void> {
    const run = this.run;
    if (!run) {
      return;
    }
    if (discard) {
      run.session?.decideOverwrite(false);
    }
    await this.stopCurrentSource(run);
    run.mode = run.resolved.mode === "playback" ? "playback" : "idle";
    this.reportRun(run);
    if (!discard) {
      void this.askToOverwriteIfPending(run);
    }
  }

  private async saveNow(): Promise<void> {
    const run = this.run;
    if (!run?.session || run.session.savingDeclined) {
      return;
    }
    if (run.session.awaitingOverwriteDecision) {
      run.session.decideOverwrite(true);
    }
    await run.saver?.flush();
    this.reportRun(run);
  }

  private seek(time: number): void {
    const media = this.run?.media ?? findPlayingMedia();
    if (media && Number.isFinite(media.duration)) {
      media.currentTime = Math.min(time, media.duration);
    }
  }

  private watchRecordingEnd(run: ActiveRun, media: HTMLMediaElement): void {
    if (run.watchingMediaEnd) {
      return;
    }
    run.watchingMediaEnd = true;
    media.addEventListener("ended", () => {
      run.session?.updateMedia(media.duration, true);
      void this.askToOverwriteIfPending(run).then(() => run.saver?.flush());
    });
    media.addEventListener("pause", () => {
      setTimeout(() => {
        if (media.paused && !media.seeking && !media.ended) {
          void this.askToOverwriteIfPending(run);
        }
      }, PAUSE_BEFORE_ASKING_MILLISECONDS);
    });
  }

  private sessionSink(run: ActiveRun, source: "asr" | "youtube-captions"): SegmentSink {
    if (run.session && run.sink) {
      run.session.useSource(source);
      return run.sink;
    }
    const { resolved } = run;
    const session = new RecognitionSession(
      {
        sessionId: crypto.randomUUID(),
        canonicalUrl: resolved.canonicalUrl,
        source,
        completionRatio: resolved.settings.completionRatio,
        placement: run.media ? "timeline" : "append",
      },
      resolved.existing,
    );
    run.session = session;
    run.saver = new SubtitleSaver(session, resolved.urlKey, resolved.revision, resolved.settings.flushIntervalSeconds, () => this.reportRun(run));
    run.saver.start();
    run.sink = (start, end, display) => {
      if (this.run !== run) {
        return;
      }
      session.recordSegment(start, end, display);
      if (run.media) {
        session.updateMedia(run.media.duration, run.media.ended);
      }
    };
    return run.sink;
  }

  private askToOverwriteIfPending(run: ActiveRun): Promise<void> {
    const session = run.session;
    if (!session?.awaitingOverwriteDecision || this.run !== run) {
      return Promise.resolve();
    }
    if (!run.overwriteQuestion) {
      run.overwriteQuestion = this.askToOverwrite(run, session).finally(() => {
        run.overwriteQuestion = null;
        this.reportRun(run);
      });
      this.reportRun(run);
    }
    return run.overwriteQuestion;
  }

  private async askToOverwrite(run: ActiveRun, session: RecognitionSession): Promise<void> {
    const confirmed = await run.overlay.askConfirmation("Merge this recording into the saved subtitles?", "Merge", "Keep existing");
    session.decideOverwrite(confirmed);
    await run.saver?.flush();
  }

  private async tryYoutubeCaptions(run: ActiveRun, media: HTMLMediaElement): Promise<boolean> {
    const captions = await enableYoutubeCaptions();
    if (!captions.available || this.run !== run) {
      return false;
    }
    const restoreCaptionButton = () => {
      if (captions.turnedOn) {
        disableYoutubeCaptions();
      }
    };
    const reader = new YoutubeCaptionReader(media, this.sessionSink(run, "youtube-captions"), () => {
      if (this.run !== run || run.stopSource !== stopReader) {
        return;
      }
      restoreCaptionButton();
      run.stopSource = null;
      run.liveCue = () => null;
      this.startMediaAsr(run, media, run.resolved.settings);
      this.reportRun(run);
    });
    const stopReader = () => {
      reader.stop();
      restoreCaptionButton();
    };
    reader.start();
    this.watchRecordingEnd(run, media);
    run.mode = "youtube-captions";
    run.liveCue = () => reader.currentCue;
    run.stopSource = stopReader;
    return true;
  }

  private startMediaAsr(run: ActiveRun, media: HTMLMediaElement, settings: ContentSettings): void {
    const sink = this.sessionSink(run, "asr");
    let partial: Cue | null = null;
    const recognizer = new StreamingRecognizer(
      media,
      {
        partial: (start, end, text) => {
          partial = { start, end, text };
        },
        final: (start, end, segments) => {
          if (partial && partial.start <= end) {
            partial = null;
          }
          sink(start, end, "");
          for (const segment of segments) {
            sink(segment.start, segment.end, segment.display);
          }
        },
        covered: (start, end) => sink(start, end, ""),
        problem: (message) => run.overlay.showMessage(message),
      },
      {
        recognizeOnlyAtNormalRate: settings.features.recognizeOnlyAtNormalRate,
        isAdPlaying: isYoutubePage() ? isYoutubeAdPlaying : () => false,
      },
    );
    run.liveCue = () => (partial && media.currentTime - partial.end <= LIVE_CAPTION_HOLD_SECONDS ? partial : null);
    try {
      recognizer.start();
      this.watchRecordingEnd(run, media);
      run.mode = "recognize";
      run.stopSource = () => recognizer.stop();
    } catch (error) {
      run.liveCue = () => null;
      const message =
        error instanceof CrossOriginMediaError
          ? "This video cannot be captured directly. Use the toolbar button → Capture tab audio."
          : `Cannot capture media audio: ${describe(error)}`;
      run.mode = "error";
      run.overlay.showMessage(message);
      this.report({ mode: "error", message });
    }
  }

  private async stopCurrentSource(run: ActiveRun): Promise<void> {
    run.stopSource?.();
    run.stopSource = null;
    run.liveCue = () => null;
    run.tabCapture?.stop();
    run.tabCapture = null;
    await run.saver?.flush();
  }

  private async beginTabCapture(): Promise<void> {
    if (this.closed || window !== window.top) {
      return;
    }
    const run = this.run ?? (await this.openRun(findPlayingMedia()));
    if (!run) {
      return;
    }
    await this.stopCurrentSource(run);
    const receiver = new TabCaptureReceiver(() => (run.media ??= findPlayingMedia()), this.sessionSink(run, "asr"));
    run.tabCapture = receiver;
    run.mode = "tab-capture";
    run.liveCue = () => (run.media ? null : receiver.latestCue);
    this.reportRun(run);
  }

  private async endTabCapture(message: string | null): Promise<void> {
    const run = this.run;
    if (!run?.tabCapture) {
      return;
    }
    await this.stopCurrentSource(run);
    run.mode = "idle";
    if (message) {
      run.overlay.showMessage(message);
    }
    this.reportRun(run);
    await this.askToOverwriteIfPending(run);
  }

  private async closeByUser(): Promise<void> {
    if (this.run) {
      await this.askToOverwriteIfPending(this.run);
    }
    this.closed = true;
    this.stopWatching?.();
    this.stopWatching = null;
    await this.teardownRun();
    this.report({ mode: "closed", message: "Captions closed on this page" });
  }

  private async teardownRun(): Promise<void> {
    const run = this.run;
    if (!run) {
      return;
    }
    run.timers.forEach(clearInterval);
    run.overlay.destroy();
    await this.stopCurrentSource(run);
    await run.saver?.stop();
    this.run = null;
  }

  private reportRun(run: ActiveRun): void {
    this.report(
      buildRunStatus({
        mode: run.mode,
        canonicalUrl: run.resolved.canonicalUrl,
        urlKey: run.resolved.urlKey,
        savedCoverage: run.resolved.existing?.coveredIntervals ?? [],
        savedCueCount: run.resolved.cues.length,
        session: run.session,
        recording: run.stopSource !== null || run.tabCapture !== null,
        askingOverwrite: run.overwriteQuestion !== null,
        media: run.media,
      }),
    );
  }

  private report(partial: Partial<TabStatus> & Pick<TabStatus, "mode">): void {
    if (window !== window.top && (partial.mode === "idle" || partial.mode === "disabled")) {
      return;
    }
    const status: TabStatus = { ...IDLE_TAB_STATUS, ...partial };
    void askBackground({ type: "report-status", status }).catch(() => undefined);
  }
}
