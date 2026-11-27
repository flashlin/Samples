# HANDOFF

## Goal

新增 `gen-pi-h3.sh`：用 pi-H3（MiniMax H3 FL2VA + pi-Flow DWD adapter）在 Mac（MPS）上從 `prompts/` 挑提示詞產生影片，操作方式與 `gen.sh`（Sulphur 3）相同。

## Current Progress

- 已定案（2026-11-27 grill）：
  - Workflow 以官方 `Lakonik/ComfyUI-piFlow` 的 `pi-H3-DWD.json` 為底（存於 `workflows/pi-H3-DWD.json`）。RunningHub 頁面需登入（API 回 403 `TOKEN_MISSION`），未取得。amao2001 的兩個 workflow 用 Kijai `SolAttnPatch`（需 Triton）與 `SelfLiftH3Sampler`，Mac 不能直接跑，只抽出其提示詞。
  - 模式只做 t2v / i2v（首幀），>15 秒拆段並以最後一格接續。
  - 預設 4 步（`--steps 2|4`），672x384、10 秒。
  - fzf 列出 `prompts/*.txt` 全部，`prompt_pih3_*` 排前面。
  - 先抽共用層：`src/video_gen_cli.py`（`ModelProfile` protocol），`sulphur_gen.py` 與 `pi_h3_gen.py` 各為一個 profile。
  - `setup.sh` 會安裝 ComfyUI-piFlow 並下載 4 步 pi-H3 模型（`SKIP_MODELS=1` 時只裝 node）。
- 已完成：程式、測試（`tests/`，17 passed）、`gen.sh` / `gen-pi-h3.sh` dry-run、ComfyUI-piFlow 載入確認（commit `4355cc4`）。
- 背景程序：模型下載 `scratchpad/tools/dl_pih3.py`，log 在 session scratchpad `dl_pih3.log`，完成訊號 `DOWNLOAD_DONE ok=True`。下載完的檔案：
  - `ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`（約 21 GB）
  - `ComfyUI/models/vae/minimax_h3_video_vae_int8_convrot.safetensors`（約 2.8 GB）
  - `ComfyUI/models/loras/pi_h3_4step_dwd_adapter_int8_convrot.safetensors`（已完成）

## Blockers and Active Decisions

- 未驗證：`int8_convrot` 底模／VAE 在 MPS 上能否執行。若失敗，先停下回報，不自行換模型。
- 不做 bf16 預先反量化快取（2026-11-27 使用者決定）：MPS 每層每次 forward 都先 dequantize 再 matmul（`comfy/ops.py:1597`），但 matmul 工作量是權重數 × token 數，佔大部分時間；轉成 bf16 會讓檔案和記憶體約翻倍（約 40 GB）。
- `BlockSparseAttention` 在非 CUDA 時自動退回 dense（`comfy_extras/nodes_sparse_attention.py` 的 `h3_eligible`），Mac 上沒有稀疏加速。
- 重構小改動：fzf 提示詞清單改為「預設檔 → 同 profile 前綴 → 其他」，gen.sh 的清單順序因此可能和以前不同。

## Next Steps

1. 等下載完成後實跑驗收：
   `./gen-pi-h3.sh --workflow t2v --prompt-file prompts/prompt_pih3_t2v.txt -y --seed 42 --resolution 672x384 --duration 10`
2. 記錄每一步的秒數與總時間，寫回本檔。
3. MPS 失敗時：保留錯誤訊息，回報使用者並提供選項。

## Worktree State

- 分支 `master`，未 commit。
- 使用者原本就有的未 commit 改動：`prompts/prompt_s3_t2v.txt`（不要動）。
- 本次改動：`gen.sh`（改成呼叫共用的 `lib/run_python_entry.sh`）、`setup.sh`、`src/sulphur_gen.py`、`src/sulphur_workflow.py`、`src/workflow_builder.py`；新增 `gen-pi-h3.sh`、`src/video_gen_cli.py`、`src/pi_h3_*.py`、`src/prompt_templates/pih3/`、`prompts/prompt_pih3_*.txt`、`workflows/pi-H3-DWD.json`、`tests/`、`plans/HANDOFF.md`。
- `.venv` 新裝 `pytest`。

## 結案（2026-11-27 21:38）

實跑驗收通過：

```
./gen-pi-h3.sh --workflow t2v --prompt-file prompts/prompt_pih3_t2v.txt -y --seed 42 --resolution 672x384 --duration 10
```

| 項目 | 數字 |
|---|---|
| 取樣 4 步 | 15:26（242 / 229 / 225 / 232 秒每步，平均 231.6 s/it） |
| ComfyUI prompt 總時間 | 16:33（含載入模型、文字編碼、DWD、音訊解碼） |
| 指令總時間（含 server 啟動） | 1003 秒（約 16.7 分鐘） |
| 輸出 | `outputs/pih3_t2v_672x384_10s_20261127_212103.mp4`，10.13 秒、672x384、24 fps、AAC 32 kHz 立體聲、2.2 MB |

- `int8_convrot` 底模在 MPS 走 emulated ops（每次 forward 先 dequantize），能跑。
- 抽第 30、200 格目視：畫面符合提示詞，顏色正常，DWD 輸出正確。
- 已知、與本次無關：comfy-cli 第一次啟動缺 `yaml`，`run_pipeline.py` 會改用備用啟動方式，能正常跑。
