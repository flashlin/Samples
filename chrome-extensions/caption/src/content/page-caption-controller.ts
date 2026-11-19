import { latestSentences, previousCue } from "../shared/cue/caption-lines";
import type { Cue, SegmentSink } from "../shared/cue/cue";
import { findCueAt, selectDisplayCue } from "../shared/cue/cue-timeline";
import type { ContentSettings, ResolvedPage, TabBroadcast, TabStatus } from "../shared/messages";
import { RecognitionSession } from "../shared/progress/recognition-session";
import { askBackground } from "./background-client";
import { CaptionOverlay } from "./caption-overlay";
import { findPlayingMedia, watchForPlayingMedia } from "./media-detector";
import { SubtitleSaver } from "./subtitle-saver";
import { TabCaptureReceiver } from "./tab-capture-receiver";
import { CrossOriginMediaError, StreamingRecognizer } from "./streaming-recognizer";
import { disableYoutubeCaptions, enableYoutubeCaptions, isYoutubeAdPlaying, isYoutubePage } from "./youtube";
import { YoutubeCaptionReader } from "./youtube-caption-reader";

const DISPLAY_REFRESH_MILLISECONDS = 200;
const LIVE_CAPTION_HOLD_SECONDS = 3;
const PAUSE_BEFORE_ASKING_MILLISECONDS = 800;

type ResolvedForCaptions = Extract<ResolvedPage, { mode: "playback" | "recognize" }>;

interface ActiveRun {
  readonly resolved: ResolvedForCaptions;
  readonly overlay: CaptionOverlay;
  readonly displayTimer: ReturnType<typeof setInterval>;
  media: HTMLMediaElement | null;
  mode: TabStatus["mode"];
  session: RecognitionSession | null;
  saver: SubtitleSaver | null;
  sink: SegmentSink | null;
  overwriteQuestion: Promise<void> | null;
  stopSource: (() => void) | null;
  liveCue: () => Cue | null;
  tabCapture: TabCaptureReceiver | null;
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
        await this.beginTabCapture();
        return;
      case "tab-capture-segment":
        this.run?.tabCapture?.receive(message.start, message.end, message.display);
        return;
      case "tab-capture-partial":
        this.run?.tabCapture?.receivePartial(message.start, message.end, message.text);
        return;
      case "tab-capture-stopped":
        await this.endTabCapture(message.message);
        return;
      case "subtitle-deleted":
        await this.dispose();
        this.closed = false;
        this.start();
        return;
    }
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
        this.startPlayback(run);
      } else if (run) {
        await this.startRecognition(run, media);
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
      displayTimer: setInterval(() => overlay.showCaption(this.displayText(run)), DISPLAY_REFRESH_MILLISECONDS),
    };
    this.run = run;
    return run;
  }

  private displayText(run: ActiveRun): string | null {
    const { current, timeline } = this.currentCue(run);
    if (!current) {
      return null;
    }
    const previous = previousCue(timeline, current, LIVE_CAPTION_HOLD_SECONDS);
    return latestSentences(previous?.text ?? null, current.text, run.resolved.settings.overlay.maxSentences);
  }

  private currentCue(run: ActiveRun): { current: Cue | null; timeline: readonly Cue[] } {
    const live = run.liveCue();
    if (live) {
      return { current: live, timeline: run.session?.liveCues ?? [] };
    }
    if (!run.media) {
      return { current: null, timeline: [] };
    }
    const time = run.media.currentTime;
    const fallbackCues = run.resolved.cues;
    if (!run.session) {
      return { current: findCueAt(fallbackCues, time), timeline: fallbackCues };
    }
    const liveCues = run.session.liveCues;
    const current = selectDisplayCue({ liveCues, liveCoverage: run.session.liveCoverage, fallbackCues, liveHoldSeconds: LIVE_CAPTION_HOLD_SECONDS }, time);
    return { current, timeline: current && liveCues.includes(current) ? liveCues : fallbackCues };
  }

  private startPlayback(run: ActiveRun): void {
    run.mode = "playback";
    this.reportRun(run);
  }

  private async startRecognition(run: ActiveRun, media: HTMLMediaElement): Promise<void> {
    const settings = run.resolved.settings;
    const youtubeStarted = isYoutubePage() && settings.features.useYoutubeCaptions && (await this.tryYoutubeCaptions(run, media));
    if (!youtubeStarted && settings.features.autoStart) {
      this.startMediaAsr(run, media, settings);
    }
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
    this.reportRun(run);
  }

  private sessionSink(run: ActiveRun, source: "asr" | "youtube-captions"): SegmentSink {
    if (run.session && run.sink) {
      run.session.useSource(source);
      return run.sink;
    }
    const { resolved } = run;
    const session = new RecognitionSession(
      { sessionId: crypto.randomUUID(), canonicalUrl: resolved.canonicalUrl, source, completionRatio: resolved.settings.completionRatio },
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
    const confirmed = await run.overlay.askConfirmation("Replace the saved subtitles with this recording?", "Replace", "Keep existing");
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
    clearInterval(run.displayTimer);
    run.overlay.destroy();
    await this.stopCurrentSource(run);
    await run.saver?.stop();
    this.run = null;
  }

  private reportRun(run: ActiveRun): void {
    this.report({
      mode: run.mode,
      canonicalUrl: run.resolved.canonicalUrl,
      urlKey: run.resolved.urlKey,
      coveredSeconds: run.session?.coveredSeconds ?? run.resolved.existing?.coveredSeconds ?? 0,
      cueCount: run.session?.liveCues.length ?? run.resolved.cues.length,
      message: this.saveStatusMessage(run),
    });
  }

  private saveStatusMessage(run: ActiveRun): string | null {
    if (run.session?.savingDeclined) {
      return "Not saving this recording: the existing subtitles were kept";
    }
    if (run.overwriteQuestion) {
      return "Waiting for your answer in the caption box";
    }
    return run.session?.awaitingOverwriteDecision ? "Not saved yet: you will be asked when you pause or stop" : null;
  }

  private report(partial: Partial<TabStatus> & Pick<TabStatus, "mode">): void {
    if (window !== window.top && (partial.mode === "idle" || partial.mode === "disabled")) {
      return;
    }
    const status: TabStatus = { canonicalUrl: null, urlKey: null, coveredSeconds: 0, cueCount: 0, message: null, ...partial };
    void askBackground({ type: "report-status", status }).catch(() => undefined);
  }
}
