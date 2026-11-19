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

Models are loaded from the Hugging Face cache. Download them once with `../setup.sh` (runs `uv run caption-server-setup`, which reads `CAPTION_ASR_MODEL` / `CAPTION_TRANSLATE_MODEL` and downloads over plain HTTP because xet downloads stall on this machine).

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
| `CAPTION_TRANSLATE_MODEL` | `mlx-community/HY-MT1.5-1.8B-8bit` | |
| `CAPTION_KEEP_TERMS` | built-in list | Comma separated terms kept in English when translating |
| `CAPTION_BILINGUAL_ENGLISH` | `true` | English lines shown as original + translation |
| `CAPTION_MAX_AUDIO_BYTES` / `CAPTION_MAX_SRT_BYTES` | `2000000` / `5000000` | |

## API

All routes require `Authorization: Bearer <token>`, a loopback `Host`, and an allowed `Origin` when one is sent.

| Method | Path | Body | Result |
|---|---|---|---|
| GET | `/health` | | engine names |
| POST | `/v1/captions/transcribe` | multipart `audio` (webm/opus, wav, …) | `text`, `language`, `translation`, `display` |
| POST | `/v1/translations` | `{"texts": [...]}` | `translations`, `displays` |
| GET | `/v1/subtitles/{urlKey}` | | `meta` + `srt`, 404 if missing |
| PUT | `/v1/subtitles/{urlKey}` | `meta`, `srt`, optional `expectedRevision` | `written`, `revision`, `reason`; 409 on revision conflict |
| DELETE | `/v1/subtitles/{urlKey}` | | 204 / 404 |

`PUT` re-checks the overwrite policy server side (same cases as `caption/test/fixtures/overwrite-policy-cases.json`), so a client cannot replace a further-along or complete subtitle.

## Test

```bash
uv run pytest -q
```
