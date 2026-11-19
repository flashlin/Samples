export const OVERLAY_HEIGHT_LINES = { min: 1, max: 10, initial: 2 } as const;

export interface OverlayLayout {
  readonly offsetX: number;
  readonly offsetY: number;
  readonly heightLines: number;
}
