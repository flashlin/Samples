export interface ChunkLimits {
  readonly minSeconds: number;
  readonly maxSeconds: number;
  readonly silenceSeconds: number;
}

export interface ChunkState {
  readonly elapsedSeconds: number;
  readonly trailingSilenceSeconds: number;
}

export function shouldCutChunk(state: ChunkState, limits: ChunkLimits): boolean {
  if (state.elapsedSeconds >= limits.maxSeconds) {
    return true;
  }
  return state.elapsedSeconds >= limits.minSeconds && state.trailingSilenceSeconds >= limits.silenceSeconds;
}

export const SILENCE_RMS_THRESHOLD = 0.01;

export function rootMeanSquare(samples: Float32Array): number {
  if (samples.length === 0) {
    return 0;
  }
  let sum = 0;
  for (const sample of samples) {
    sum += sample * sample;
  }
  return Math.sqrt(sum / samples.length);
}
