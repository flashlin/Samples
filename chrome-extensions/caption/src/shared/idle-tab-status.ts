import type { TabStatus } from "./messages";

export const IDLE_TAB_STATUS: TabStatus = {
  mode: "idle",
  canonicalUrl: null,
  urlKey: null,
  coveredSeconds: 0,
  cueCount: 0,
  message: null,
  recording: false,
  canSave: false,
  mediaDuration: null,
  currentTime: null,
  coverage: [],
};
