import { compareCuesByStart, type Cue } from "./cue";

function overlaps(a: Cue, b: Cue): boolean {
  return a.start < b.end && b.start < a.end;
}

export function mergeCues(existing: readonly Cue[], incoming: readonly Cue[]): Cue[] {
  const kept = existing.filter((cue) => !incoming.some((added) => overlaps(cue, added)));
  return [...kept, ...incoming].sort(compareCuesByStart);
}
