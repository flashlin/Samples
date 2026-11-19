# HANDOFF — caption Chrome extension

> 最後更新：2026-11-19

## Goal

建立 `chrome-extensions/caption/`：TS Chrome extension + 本機 Python caption-server。網頁語音即時辨識（台灣口音、中英混合），tooltip 顯示，存 `.srt`；同 URL 已完成就直接讀 `.srt`。完整計畫：`plans/caption-speech-to-srt-extension.md`（§0 是使用者已決定事項）。

## Current Progress

- 計畫：`plans/caption-speech-to-srt-extension.md`。§0 = 使用者決定；§16 = 實作紀錄（以 §16 為準）。
- 架構已改為**串流辨識**（2026-11-19 晚）：content ScriptProcessor 16kHz PCM → background Port → 一條 WebSocket `/v1/captions/stream` → server（Silero VAD 切句、說話中每 0.5 秒 partial、句尾 final = aligner + 標點 + 翻譯）。tab capture 走同一條 relay。舊的 MediaRecorder 分段與 HTTP 轉錄端點已刪除。
- 覆蓋規則：頁面已有 .srt 時，錄製停止（暫停 / 播完 / 停止擷取 / 按 ✕）後在字幕框詢問 Replace / Keep existing。
- 字幕：單行（目前這句）、可拖曳、常駐框。
- 驗證：`cd caption && pnpm check`（unit 120）、`cd caption/server && uv run pytest -q`（77）、`pnpm e2e`、`pnpm e2e:youtube` 通過；延遲量測 `node test/e2e/caption-latency.diagnose.mjs`（YouTube 前 34 秒，延遲 0.8–1.8 秒）。
- 沒有背景程序在跑。上次 commit：`215f42e0e`；之後所有改動未 commit。

## Blockers and Active Decisions

- YouTube 字幕讀畫面 DOM；不用 YouTube 內部 API（auto mode 擋過）。
- 根目錄 `.gitignore` 有 `*.lock`；使用者決定 `caption/server/uv.lock` 要進版控，例外寫在 `caption/.gitignore`（`!server/uv.lock`）。
- 只能手動驗證：Capture tab audio（Meet）、有字幕的 YouTube 影片、跨源影片提示。

## Next Steps

1. 使用者重跑 `caption/setup.sh`（新電腦才會真的下載 aligner 與標點模型）、重新載入 extension、重啟 `dev.sh`，實測。
2. VAD 切段已在串流架構中完成。
3. 待釐清：使用者 18:14 錄到 37 秒停止的原因（當時是舊架構；請使用者用新版重測並描述畫面）。
3. 依回饋修正；這一輪改動待使用者要求再 commit。

## Worktree State

- 分支 `master`；最後一次 commit 是 `215f42e0e`。之後的改動（單行字幕、句子分段、sherpa-onnx 依賴、詢問覆蓋）尚未 commit。`../minimax/prompt.txt` 是先前就存在的未 commit 改動，與本工作無關，未動。
