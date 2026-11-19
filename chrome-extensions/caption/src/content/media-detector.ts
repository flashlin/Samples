function visibleArea(element: HTMLMediaElement): number {
  const rect = element.getBoundingClientRect();
  return Math.max(0, rect.width) * Math.max(0, rect.height);
}

function isPlaying(element: HTMLMediaElement): boolean {
  return !element.paused && !element.ended && element.readyState > HTMLMediaElement.HAVE_CURRENT_DATA;
}

export function findPlayingMedia(root: ParentNode = document): HTMLMediaElement | null {
  const playing = [...root.querySelectorAll<HTMLMediaElement>("video, audio")].filter(isPlaying);
  return playing.sort((a, b) => visibleArea(b) - visibleArea(a))[0] ?? null;
}

export function watchForPlayingMedia(onPlay: (element: HTMLMediaElement) => void): () => void {
  const handler = (event: Event) => {
    if (event.target instanceof HTMLMediaElement) {
      onPlay(event.target);
    }
  };
  document.addEventListener("playing", handler, true);
  const alreadyPlaying = findPlayingMedia();
  if (alreadyPlaying) {
    queueMicrotask(() => onPlay(alreadyPlaying));
  }
  return () => document.removeEventListener("playing", handler, true);
}
