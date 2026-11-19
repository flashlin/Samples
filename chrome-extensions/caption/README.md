# caption

Chrome extension (MV3) that shows live captions for web media, saves them as `.srt` per page URL on a local caption-server, and replays the saved `.srt` on later visits.

- Speech recognition: Taiwanese Mandarin / English code-switching (`JacobLinCool/TEA-ASR-1.1-mini`), output in Traditional Chinese.
- Streaming: the page audio is sent as 16 kHz PCM over one WebSocket (through the background worker) to caption-server, which cuts sentences with Silero VAD, re-transcribes the sentence in progress every 0.5 s for interim captions, and finalises each sentence with punctuation restoration (sherpa-onnx CT-Transformer) and per-character timestamps (`Qwen/Qwen3-ForcedAligner-0.6B`). Captions trail speech by about 1 s; the overlay shows one sentence at a time.
- English sentences are shown bilingual (original + Traditional Chinese translation); mixed sentences keep their English words.
- YouTube videos that have captions use the captions shown by the player instead of ASR.
- Pages whose media cannot be captured directly (cross-origin video, Google Meet, …): toolbar button → **Capture tab audio**.

## Setup

```bash
cd caption
./setup.sh

cd caption/server
./dev.sh

cd caption
./build.sh
```

1. `chrome://extensions` → Developer mode → **Load unpacked** → `caption/dist`. `dev.sh` already allows this unpacked extension's id.
2. Extension options → paste the token from `~/.caption-server/token` → **Save** → **Test connection**.

## Scripts

| Command | What it does |
|---|---|
| `./setup.sh` | `uv sync` for the server and download the ASR / translation models named in the server settings into the Hugging Face cache (models are not stored in this repo) |
| `./build.sh` | `pnpm install --frozen-lockfile` + `pnpm build`; output in `caption/dist/` |
| `pnpm build` | Bundle `src/` into `dist/` with esbuild |
| `pnpm test` | Unit tests (vitest) |
| `pnpm typecheck` | `tsc --noEmit` |
| `pnpm e2e:youtube` | Same harness against a real YouTube video without captions (automated playback is blocked by YouTube after about 40 s) |
| `node test/e2e/caption-latency.diagnose.mjs` | Measures how long after each spoken sentence the caption appears on a YouTube video |
| `pnpm e2e` | Builds, starts the real caption-server, loads the extension in Playwright Chromium, plays a `say`-generated speech clip and checks the saved SRT and the replay path (macOS, needs the models in the Hugging Face cache and `node node_modules/playwright-core/cli.js install chromium` once) |

## Layout

| Path | Role |
|---|---|
| `src/shared/` | Pure logic shared by every context: SRT codec, URL canonicalizer, coverage, overwrite policy, recognition session, chunk cut policy, config schema, message types |
| `src/background/` | Only place that holds the server token and talks to caption-server; page resolution, tab status badge, tab capture coordination |
| `src/content/` | Media detection, audio tap, YouTube caption reader, overlay with close button, subtitle saver |
| `src/offscreen/` | Tab audio capture for the **Capture tab audio** button |
| `src/popup/`, `src/options/` | Toolbar popup and options page |
| `server/` | caption-server (Python, see `server/README.md`) |
