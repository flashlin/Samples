import type { CanonicalizerOptions } from "./url-canonicalizer";

const YOUTUBE_ORIGIN = "https://www.youtube.com";
const YOUTUBE_WATCH_PATH = "/watch";
const YOUTUBE_VIDEO_PARAM = "v";

export const DEFAULT_CANONICALIZER_OPTIONS: CanonicalizerOptions = {
  stripParams: ["utm_*", "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid", "si", "ref", "ref_src", "spm"],
  siteRules: [
    {
      hosts: ["www.youtube.com", "youtube.com", "m.youtube.com", "www.youtube-nocookie.com"],
      canonicalOrigin: YOUTUBE_ORIGIN,
      keepParams: [YOUTUBE_VIDEO_PARAM],
      pathAliases: [
        { pattern: "^/shorts/([^/?#]+)", path: YOUTUBE_WATCH_PATH, param: YOUTUBE_VIDEO_PARAM },
        { pattern: "^/live/([^/?#]+)", path: YOUTUBE_WATCH_PATH, param: YOUTUBE_VIDEO_PARAM },
        { pattern: "^/embed/([^/?#]+)", path: YOUTUBE_WATCH_PATH, param: YOUTUBE_VIDEO_PARAM },
      ],
    },
    {
      hosts: ["youtu.be"],
      canonicalOrigin: YOUTUBE_ORIGIN,
      keepParams: [YOUTUBE_VIDEO_PARAM],
      pathAliases: [{ pattern: "^/([^/?#]+)$", path: YOUTUBE_WATCH_PATH, param: YOUTUBE_VIDEO_PARAM }],
    },
    {
      hosts: ["www.bilibili.com", "bilibili.com", "m.bilibili.com"],
      canonicalOrigin: "https://www.bilibili.com",
      keepParams: ["p"],
      pathAliases: [],
    },
  ],
};
