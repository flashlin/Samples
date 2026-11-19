# caption-server

Local ASR, English → Traditional Chinese translation, and SRT storage for the `caption` Chrome extension.

## Run

```bash
cd caption/server
./dev.sh
```

`dev.sh` points `CAPTION_DEV_EXTENSION_DIR` at `caption/dist`; the server computes the unpacked extension id from that path (same rule Chrome uses: SHA-256 of the absolute path, first 32 hex digits mapped to `a`–`p`) and allows `chrome-extension://<id>`. The startup log prints the computed origin. Load the extension from exactly `caption/dist`, otherwise the id differs.

For a packed or store-installed extension, set the origin explicitly instead:

```bash
CAPTION_ALLOWED_ORIGINS=chrome-extension://<extension-id> uv run caption-server
```

On first start a bearer token is generated at `~/.caption-server/token` (mode 600). Paste it into the extension options page.

Models are loaded from the Hugging Face cache. Download them once with `../setup.sh` (runs `uv run caption-server-setup`, which reads `CAPTION_ASR_MODEL`, `CAPTION_ALIGNER_MODEL`, `CAPTION_PUNCTUATION_MODEL` and `CAPTION_TRANSLATE_MODEL` and downloads over plain HTTP because xet downloads stall on this machine).

## Settings (env, prefix `CAPTION_`)

| Variable | Default | Notes |
|---|---|---|
| `CAPTION_DATA_DIR` | `~/.caption-server` | `subtitles/<urlKey>.srt`, `<urlKey>.meta.json`, `<urlKey>.srt.prev`, `audit.jsonl`, `token` |
| `CAPTION_HOST` / `CAPTION_PORT` | `127.0.0.1` / `8765` | Keep loopback unless TLS and per-user tokens are in place |
| `CAPTION_TOKEN` | generated | Bearer token required on every request |
| `CAPTION_ALLOWED_ORIGINS` | empty | Comma separated; requests carrying any other `Origin` are rejected |
| `CAPTION_DEV_EXTENSION_DIR` | unset | Unpacked extension directory; its computed `chrome-extension://<id>` is added to the allowed origins (`dev.sh` sets it) |
| `CAPTION_ASR_MODEL` | `JacobLinCool/TEA-ASR-1.1-mini` | Any Qwen3-ASR family checkpoint, e.g. `JacobLinCool/TEA-ASR-1.1` |
| `CAPTION_ASR_LANGUAGE` | `Chinese` | Language hint; English speech still comes out as English |
| `CAPTION_ASR_DEVICE` | `mps` | `cpu` also works |
| `CAPTION_ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | Per-character timestamps so each sentence gets its own start/end |
| `CAPTION_PUNCTUATION_MODEL` | `ranger810/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8` | Restores punctuation so run-on ASR text can be split into sentences |
| `CAPTION_MAX_SENTENCE_UNITS` | `24` | Longer sentences are split at commas into pieces of at most this many characters / words |
| `CAPTION_TRANSLATE_MODEL` | `mlx-community/HY-MT1.5-1.8B-8bit` | |
| `CAPTION_KEEP_TERMS` | built-in list | Comma separated terms kept in English when translating |
| `CAPTION_BILINGUAL_ENGLISH` | `true` | English lines shown as original + translation |
| `CAPTION_MAX_SRT_BYTES` | `5000000` | |
| `CAPTION_VAD_MODEL` | `istupakov/silero-vad-onnx` | Silero VAD used to cut the live stream into sentences |
| `CAPTION_VAD_THRESHOLD` / `CAPTION_VAD_MIN_SILENCE_SECONDS` / `CAPTION_VAD_MAX_SPEECH_SECONDS` | `0.5` / `0.3` / `10` | Speech probability threshold, pause that ends a sentence, forced cut for long speech |
| `CAPTION_STREAM_PARTIAL_INTERVAL_SECONDS` | `0.5` | How often the sentence in progress is re-transcribed for interim captions |

## API

HTTP routes require `Authorization: Bearer <token>`; the WebSocket sends the token in its first message (browsers cannot set headers on WebSockets). Every request needs a loopback `Host` and an allowed `Origin` when one is sent.

| Method | Path | Body | Result |
|---|---|---|---|
| GET | `/health` | | engine names |
| WS | `/v1/captions/stream` | first text message `{"type":"start","token":…}`; then binary frames (8-byte little-endian float64 media start time + 16 kHz mono int16 PCM) and text `{"type":"break"}` / `{"type":"stop"}` | `ready`, `partial` (`start`, `end`, `text`; every 0.5 s while someone speaks), `final` (`start`, `end`, `segments[]` with per-sentence `start`/`end`/`display`), `covered` (`start`, `end`), `error` |
| POST | `/v1/translations` | `{"texts": [...]}` | `translations`, `displays` |
| GET | `/v1/subtitles/{urlKey}` | | `meta` + `srt`, 404 if missing |
| PUT | `/v1/subtitles/{urlKey}` | `meta`, `srt`, optional `expectedRevision`, `confirmOverwrite` | `written`, `revision`, `reason` (`confirmation-required` when another session's subtitle exists and `confirmOverwrite` is false); 409 on revision conflict |
| DELETE | `/v1/subtitles/{urlKey}` | | 204 / 404 |

`PUT` re-checks the overwrite policy server side (same cases as `caption/test/fixtures/overwrite-policy-cases.json`): a subtitle saved by another session is only replaced when the request carries `confirmOverwrite: true`, which the extension sends after the user clicks **Replace** in the caption box.

## Test

```bash
uv run pytest -q
```
