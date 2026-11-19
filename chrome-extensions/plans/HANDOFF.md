# HANDOFF — caption Chrome extension

> 最後更新：2026-11-19

## Goal

建立 `chrome-extensions/caption/`：TS Chrome extension + 本機 Python caption-server。網頁語音即時辨識（台灣口音、中英混合），tooltip 顯示，存 `.srt`；同 URL 已完成就直接讀 `.srt`。完整計畫：`plans/caption-speech-to-srt-extension.md`（§0 是使用者已決定事項）。

## Current Progress

- 計畫：`plans/caption-speech-to-srt-extension.md`。§0 = 使用者決定；§16 = 實作紀錄與偏離原計畫之處（以 §16 為準）。
- extension + caption-server 已完成並 commit 在分支 `caption-extension`。使用說明：`caption/README.md`、`caption/server/README.md`。
- 安裝流程：`caption/setup.sh`（uv sync + 下載模型到 HF cache）→ `caption/build.sh`（打包到 `caption/dist`）→ `caption/server/dev.sh`（啟動 server，自動允許 `caption/dist` 的 extension id）。
- 最近一輪（2026-11-19 16:50）：兩行滾動字幕、可拖曳且常駐的 tooltip、暫時字幕（約 1–2 秒延遲）、YouTube 無字幕影片自動改走 ASR。
- 驗證：`cd caption && pnpm check`（unit 122）、`cd caption/server && uv run pytest -q`（61）、`pnpm e2e`、`pnpm e2e:youtube` 全過。
- 沒有背景程序在跑（使用者自己的 `dev.sh` 可能在 8765）。

## Blockers and Active Decisions

- YouTube 字幕讀畫面 DOM；不用 YouTube 內部 API（auto mode 擋過）。
- 根目錄 `.gitignore` 有 `*.lock`，所以 `caption/server/uv.lock` 沒有進版控；要不要加例外由使用者決定。
- 只能手動驗證：Capture tab audio（Meet）、有字幕的 YouTube 影片、跨源影片提示。

## Next Steps

1. 使用者重新載入 extension（`chrome://extensions` ↻）並重啟 `dev.sh`，實測。
2. 依回饋修正。
3. 分支 `caption-extension` 要不要 merge 回 `master` 由使用者決定。

## Worktree State

- 分支 `caption-extension`；`caption/`、`plans/` 已 commit。`../minimax/prompt.txt` 是先前就存在的未 commit 改動，與本工作無關，未動。
