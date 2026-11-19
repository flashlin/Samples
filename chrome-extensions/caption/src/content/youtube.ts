const YOUTUBE_HOST_PATTERN = /(^|\.)youtube(-nocookie)?\.com$/;
const PLAYER_SELECTOR = "#movie_player, .html5-video-player";
const SUBTITLES_BUTTON_SELECTOR = ".ytp-subtitles-button";
const CAPTION_SEGMENT_SELECTOR = ".ytp-caption-segment";
const AD_SHOWING_CLASS = "ad-showing";
const CAPTION_TOGGLE_SETTLE_MILLISECONDS = 1000;
const HIDE_NATIVE_CAPTIONS_CSS = ".ytp-caption-window-container { opacity: 0 !important; }";

export function isYoutubePage(location: Location = window.location): boolean {
  return YOUTUBE_HOST_PATTERN.test(location.hostname);
}

export function isYoutubeAdPlaying(): boolean {
  return document.querySelector(PLAYER_SELECTOR)?.classList.contains(AD_SHOWING_CLASS) ?? false;
}

function subtitlesButton(): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>(SUBTITLES_BUTTON_SELECTOR);
}

export interface YoutubeCaptionState {
  readonly available: boolean;
  readonly turnedOn: boolean;
}

export async function enableYoutubeCaptions(): Promise<YoutubeCaptionState> {
  const button = subtitlesButton();
  if (!button) {
    return { available: false, turnedOn: false };
  }
  if (button.getAttribute("aria-pressed") === "true") {
    return { available: true, turnedOn: false };
  }
  button.click();
  await new Promise((resolve) => setTimeout(resolve, CAPTION_TOGGLE_SETTLE_MILLISECONDS));
  const available = button.getAttribute("aria-pressed") === "true";
  return { available, turnedOn: available };
}

export function disableYoutubeCaptions(): void {
  const button = subtitlesButton();
  if (button?.getAttribute("aria-pressed") === "true") {
    button.click();
  }
}

export function readVisibleYoutubeCaption(): string {
  return [...document.querySelectorAll(CAPTION_SEGMENT_SELECTOR)]
    .map((segment) => segment.textContent?.trim() ?? "")
    .filter(Boolean)
    .join("\n");
}

export function hideNativeYoutubeCaptions(): () => void {
  const style = document.createElement("style");
  style.setAttribute("data-caption-extension", "");
  style.textContent = HIDE_NATIVE_CAPTIONS_CSS;
  document.documentElement.append(style);
  return () => style.remove();
}
