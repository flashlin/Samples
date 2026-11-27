# HANDOFF

## Goal

無進行中目標。上一個目標（`gen-pi-h3.sh`）已結案，見 `docs/handoff-history-2026-11-27-pi-h3-gen.md`。

## Current Progress

- `gen-pi-h3.sh` 已完成並實跑驗收（Mac，672x384、10 秒、4 步約 16.7 分鐘）。
- 沒有背景程序。

## Blockers and Active Decisions

- 不做 bf16 預先反量化快取（使用者決定，理由見歷史檔）。

## Next Steps

1. 無。

## Worktree State

- 分支 `master`，本次改動已 commit 並 push（`586b8f364`）。
- 使用者原本就有的未 commit 改動：`prompts/prompt_s3_t2v.txt`（不要動）。
- `.venv` 新裝 `pytest`。
