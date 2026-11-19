export interface Cue {
  readonly start: number;
  readonly end: number;
  readonly text: string;
}

export function compareCuesByStart(a: Cue, b: Cue): number {
  return a.start - b.start || a.end - b.end;
}
