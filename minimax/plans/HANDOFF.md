# HANDOFF

## Goal

無進行中目標。上一個目標（`gen-pi-h3.sh`）已結案，見 `docs/handoff-history-2026-11-27-pi-h3-gen.md`。

## Current Progress

- `gen-pi-h3.sh` 已完成並實跑驗收（Mac，672x384、10 秒、4 步約 16.7 分鐘）。
- 沒有背景程序。

## Blockers and Active Decisions

- 不做 bf16 預先反量化快取（使用者決定，理由見歷史檔）。

## Next Steps

1. 等使用者決定是否 commit。

## Worktree State

- 分支 `master`，未 commit。
- 使用者原本就有的未 commit 改動：`prompts/prompt_s3_t2v.txt`（不要動）。
- 本次改動：`gen.sh`、`setup.sh`、`src/sulphur_gen.py`、`src/sulphur_workflow.py`、`src/workflow_builder.py`；新增 `gen-pi-h3.sh`、`lib/run_python_entry.sh`、`src/video_gen_cli.py`、`src/pi_h3_assets.py`、`src/pi_h3_gen.py`、`src/pi_h3_workflow.py`、`src/prompt_templates/pih3/`、`prompts/prompt_pih3_*.txt`、`workflows/pi-H3-DWD.json`、`tests/`、`plans/HANDOFF.md`、`docs/`。
- `.venv` 新裝 `pytest`。
