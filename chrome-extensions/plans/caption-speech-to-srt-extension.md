# caption：網頁語音即時辨識 + SRT 快取 Chrome Extension 計畫

> 狀態：Open Questions Q1–Q9 已全部由使用者回覆（2026-11-19，見 §0）；Phase 0–6 已實作；自動測試全綠；待使用者手動驗證 YouTube 字幕與 tab capture（見 §16）
> 位置：`chrome-extensions/caption/`（新建）
> 日期：2026-11-19

---

## 0. 使用者已決定事項（2026-11-19，優先於下文所有「暫定」）

| # | 決定 |
|---|---|
| 架構 | 本機 Python caption-server 做 ASR + 存真實 `.srt`（ASR-A + 儲存-A） |
| ASR 模型 | 要能中英混合、台灣口音、要快；找到台灣模型就下載來用 → TEA-ASR-1.1(-mini)，本機實測後定預設 |
| Q2 | 舊 partial 還沒被超過前，overlay **也顯示舊 partial 的字幕**（新的還沒到的時間段用舊的） |
| Q3 | ~~覆蓋條件：新涵蓋秒數 > 舊涵蓋秒數~~ → **2026-11-19 改為：頁面已有 .srt（partial 或 complete）時，錄製後在字幕框直接詢問是否覆蓋，不再依長短自動覆蓋** |
| Q4 | 中文一律輸出**繁體** |
| Q5 | **要支援** Google Meet 等 WebRTC 會議（走 tabCapture，需點擊啟用） |
| Q6 | **所有網站**自動啟用；tooltip 視窗要有**關閉按鈕** |
| Q7 | YouTube 有字幕就**直接用既有字幕**，不跑 ASR；英文字幕 → 雙語（英文原文 + 繁中翻譯） |
| Q8 | 中英混合句**保留英文**，不翻 |
| Q9 | 英文句輸出**雙語**（英文原文 + 繁中翻譯） |

---

## 1. 需求重述與解讀

原始需求：

1. 建立 `caption/` 資料夾，建立 TypeScript 專案，產出 Chrome Extension。
2. 啟動時監聽網頁上的語音。
3. 檢查該網頁 URL 是否曾經做過語音辨識：
   - **沒做過** → 即時語音辨識，用 tooltip 小視窗顯示字幕，並存成 `.srt`。
   - **做到一半的舊檔** → 當「這次的辨識進度」超過「舊檔的進度」時，直接覆蓋舊檔。
   - **做過（完整）** → 直接讀 `.srt` 顯示，不再辨識。

### 1.1 需要釐清的多種解讀（全部列出，未默默選一個）

| 項目 | 解讀 A | 解讀 B | 本計畫暫定 |
|---|---|---|---|
| 「監聽網頁上的語音」 | 擷取頁面 `<video>/<audio>` 播放的聲音 | 擷取整個分頁輸出的聲音（含 WebRTC 會議、Web Audio） | A 為主，B 用於會議（已決定要支援，需點擊） |
| 「啟動時」 | 每個網頁一載入就自動跑 | 只在使用者允許的網站（allowlist）自動跑 | **已決定 A**（所有網站；保留 denylist 設定，預設空） |
| 「曾經辨識過」 | 有任何紀錄就算 | 只有 `status=complete` 才算 | B；`partial` 視為「沒跑完」，要重跑 |
| 「進度超過一半的檔案就覆蓋」 | 新 session 的已涵蓋秒數 > 舊檔已涵蓋秒數 → 覆蓋 | 新 session 進度 > 媒體總長 50% → 覆蓋 | **已決定 A** |
| `.srt` 存在哪 | 使用者電腦上的真實檔案 | 瀏覽器內部儲存（IndexedDB），需要時匯出 | **已決定 A**：caption-server 磁碟 |

### 1.2 重要的技術事實（影響設計，先講清楚）

- **Web Speech API（`webkitSpeechRecognition`）只能聽麥克風**，不能吃分頁音訊或 `MediaStream`。所以它不能拿來辨識網頁播放的聲音。這條是硬性阻礙，寫在方案比較表裡。
- **Extension 不能讀回 Downloads 資料夾的檔案**。`chrome.downloads` 只能寫、不能讀。所以「存 `.srt` 到下載資料夾，下次再讀回來」做不到；要嘛存在 extension 內部（IndexedDB），要嘛存在一個本機/遠端 server。
- **`chrome.tabCapture` 需要使用者手勢**（點 extension 圖示），不能「頁面一載入就自動開始」。
- **`HTMLMediaElement.captureStream()`** 可以在 content script 自動擷取，不需手勢，且時間軸天然對齊 `video.currentTime`；但跨網域且沒 CORS 的媒體、DRM（EME，例如 Netflix）可能拿到靜音或例外，需 Phase 0 實測。
- **SRT 時間軸要用「媒體時間」（`currentTime`），不是牆上時鐘**。使用者暫停、拖曳、改播放速度時，才不會錯位。
- **Chrome 內建 Live Caption** 可以即時顯示字幕，但不會存 `.srt`、不能重播舊字幕，不符合需求；只列為「如果只要看不要存」的更簡單替代。

---

## 2. Goals（目標）

1. `caption/` 是獨立的 TypeScript + pnpm 專案，build 出 Manifest V3 extension（unpacked 可載入）。
2. 在允許的網站上，偵測到媒體播放時自動判斷：
   - 有完整字幕 → **播放模式**：依 `currentTime` 顯示既有 `.srt`。
   - 沒有或只有部分字幕 → **辨識模式**：即時辨識、tooltip 顯示、持續寫入 `.srt`。
3. 部分字幕的覆蓋規則可預期、可測試、不會因為多分頁或重試把好資料蓋成壞資料。
4. ASR 引擎、字幕儲存、URL 正規化規則都是**可替換的介面**，不寫死在流程裡。
5. 所有核心邏輯（SRT 編解碼、URL 正規化、進度計算、覆蓋判斷、cue 查找）是純函式，有單元測試。

## 3. Non-goals（不做）

- 說話者分辨（diarization）、翻譯、字幕編輯 UI。
- DRM 保護內容（EME）的擷取。
- Firefox / Safari 支援（介面預留，MVP 不做）。
- 直播（無固定長度）的「完成」判定：MVP 只會一直是 `partial`。
- 多人共用、雲端同步（介面預留，MVP 單機）。

---

## 4. 方案比較（先列表，再推薦）

### 4.1 音訊擷取方式

| 方案 | 優點 | 缺點 |
|---|---|---|
| **A. `HTMLMediaElement.captureStream()`（content script）** | 不需手勢，能真正「自動開始」；時間軸直接對齊 `currentTime`；只抓媒體聲音，不會混到通知音 | 跨網域無 CORS 的媒體、DRM 可能靜音或拋例外；抓不到 WebRTC 會議、純 Web Audio 的聲音；一頁多個媒體要挑一個 |
| B. `chrome.tabCapture` + offscreen document | 抓得到分頁所有聲音（含會議） | **需要點擊手勢，無法自動開始**；要另外對時（分頁音訊沒有 `currentTime`）；擷取後原分頁會靜音，需在 offscreen 重新播放 |
| C. Web Speech API | 免 server、免費 | **硬性阻礙**：只能聽麥克風，無法吃分頁音訊；靠喇叭→麥克風回授品質很差；不給時間戳，難產 SRT |

**推薦**：A 為主，B 做成「點擊圖示手動啟用」的 fallback（Phase 5）。C 不採用（證據：API 不接受 MediaStream 輸入）。

### 4.2 ASR 引擎

| 方案 | 優點 | 缺點 |
|---|---|---|
| **A. 本機 ASR server（OpenAI 相容 `/v1/audio/transcriptions`，後端用 mlx-whisper 或 Qwen3 ASR）** | 你的機器已有 mlx-whisper / Qwen3 ASR（`titan-asr`、`tool-youtube` skill 有用過）；不需雲端 key；音訊不離開本機；之後換雲端只換 endpoint | 要多一個常駐程序；不是純 TS（server 會是 Python）；server 沒開時 extension 只能播放、不能辨識 |
| B. 雲端 API（OpenAI / Deepgram / Google STT）經由 proxy | 準確、延遲低、零本機負擔 | 要錢；音訊外送有隱私風險；**API key 不能放 extension**，一定要自架 proxy |
| C. 瀏覽器內 Whisper（transformers.js + WebGPU，在 offscreen document 跑） | 完全不用 server，純 TS | 模型 40–150MB 下載；速度看 GPU，即時性不穩；中文準確度低於 large 模型 |

**推薦**：A。理由：重用現有本機 ASR、資料不外送、介面與 B 相同（之後換雲端成本低）。

### 4.2.1 ASR 模型選擇（2026-11-19 追加需求：中英混合、台灣口音、要快）

公開數據（同一份評測協定，來源：TEA-ASR-1.1 model card，作者自測；MER %，越低越好）：

| 模型 | 大小 / 底座 | CV19 zh-TW | ASCEND（中英混） | CSZS（中英混） | NTUML2021（台灣課堂中英混） | 輸出字體 | Apple Silicon |
|---|---|---|---|---|---|---|---|
| **TEA-ASR-1.1** | 2B / Qwen3-ASR-1.7B fine-tune | **3.58** | **9.60** | **10.94** | **6.67** | 繁體 + 台灣用語（模型直接輸出） | 頁面沒寫；`qwen-asr`（torch），需實測 MPS 或轉 MLX |
| TEA-ASR-1.1-mini | 0.8B / Qwen3-ASR-0.6B | 5.12 | 11.20 | 12.51 | 7.53 | 繁體 + 台灣用語 | 同上，較快 |
| Qwen3-ASR-1.7B | 1.7B | 3.90 | 10.57 | 11.03 | 10.12 | 需 OpenCC 轉繁 | `speech` CLI（MLX，`titan-asr` 用過） |
| Breeze-ASR-25 | 2B / Whisper-large-v2 | 8.03 | 17.53 | 12.18 | 7.50 | 繁體 | 有社群 MLX 版（Q5 1.07GB，M4 Pro RTFx 17.7x） |
| Whisper-large-v3 | 1.5B | 10.17 | 19.61 | 23.24 | 9.68 | 常出簡體 | mlx-whisper |
| TingMix-Large v3 | ~0.3B encoder + CTC | 12.02 | 20.02 | — | 13.39（in-domain） | 未說明 | 頁面偏 GPU；Small 版 CPU 原生串流 |

注意：上表是 model card 作者自測，不是獨立評測；速度只有 Breeze MLX 有公開數字。**必須在本機（M4 Max）用真實台灣中英混合語音實測**準確度與每段延遲後才定案（Phase 0 probe C）。

**已實測定案（見 Phase 0 probe C）**：預設 TEA-ASR-1.1-mini；TEA-ASR-1.1、Breeze-ASR-25 MLX Q5 為設定選項。模型名稱做成 `asr.model` 設定，不寫死。三個模型都已下載到 `~/.cache/huggingface/hub`。

### 4.2.2 英文 → 繁體中文翻譯（2026-11-19 追加需求）

| 方案 | 優點 | 缺點 |
|---|---|---|
| **A. ASR 之後接本機 MT（caption-server 內用 mlx-lm 跑 `HY-MT1.5-1.8B`；`Translator` 介面也可接 OpenAI 相容 endpoint）** | 與 ASR 分離，可單獨換模型；保留原文；只翻「英文為主的句子」 | 多一段延遲（每句一次 LLM 呼叫）；小模型偶爾把指令翻進輸出 |
| B. 讓 ASR 模型直接「翻譯式輸出」（Qwen3/TEA 在 `language="Chinese"` 下有時會把英文翻成中文） | 零額外延遲 | **不可控**：TEA card 明講這是要避免的副作用，並提供 `keep-en` tag 防止；無法保證英文句一定被翻 |
| C. Whisper `task=translate` | 內建 | 只能翻成**英文**，方向相反，不符合需求 |

推薦 A。流程：`ASR segment` → `LanguageClassifier`（英文字元比例 ≥ 閾值才算英文句）→ 英文句送 `Translator` → 中文句只做 OpenCC `s2tw` 正規化（不用 `s2twp`，它會改詞造成誤譯；TEA mini 實測仍會漏少數簡體字，所以不能跳過）。
中英混合句（例：「這個 API 很 slow」）**預設不翻**，保留技術詞；是否也要翻見 Open Question Q8。
SRT 存「顯示文字」，meta 另存原文，避免翻譯錯了就找不回原文。

### 4.3 字幕（`.srt`）儲存位置

| 方案 | 優點 | 缺點 |
|---|---|---|
| **A. 存在本機 caption-server 的磁碟（`<dataDir>/<urlKey>.srt` + `<urlKey>.meta.json`）** | 真正的 `.srt` 檔，能被其他工具讀；extension 重裝不會遺失；能讀回來（符合「直接抓取 .srt」字面需求）；與 ASR 方案 A 同一個程序 | 依賴 server；server 要做路徑安全、CAS 覆蓋 |
| B. IndexedDB（extension 內）+ `chrome.downloads` 匯出 `.srt` | 不用 server 也能播放；實作最少 | 不是真檔案；extension 移除就消失；下載出去的檔案讀不回來 |
| C. File System Access API（使用者選資料夾） | 真檔案、不用 server | 每次重開瀏覽器都要使用者手勢重新授權；content script 不能直接用，要經 extension page；最複雜 |

**推薦**：若 ASR 選 A → 儲存選 A，並保留 `SubtitleStore` 介面讓 B 可以當 cache 或離線替代。
若你希望「只要一個 TS 專案、不要 server」→ 改走 ASR-C + 儲存-B（見 Open Questions Q1）。

### 4.4 build 工具

| 方案 | 優點 | 缺點 |
|---|---|---|
| **A. esbuild（每個 entry 各自 bundle 成 IIFE/ESM）** | content script 必須是單檔非 module，esbuild 最直接；設定最少；快 | 沒有 HMR |
| B. Vite 多 entry（同 `atool/`） | 與現有專案一致 | Rollup 會切 shared chunk，content script 不能 `import`，要額外處理（`atool` 用 post-build script 補） |
| C. Vite + `@crxjs/vite-plugin` | 有 HMR、manifest 自動處理 | 多一個第三方依賴；MV3 + 最新 Vite 相容性要再確認 |

**推薦**：A。UI 很小（overlay、popup、options），不需要 Vue。

---

## 5. Assumptions and constraints（假設與限制）

- 使用者用 Chrome 116+（offscreen、`tabCapture.getMediaStreamId` 在 service worker 可用）。
- macOS Apple Silicon，本機可跑 mlx-whisper / Qwen3 ASR。
- 套件管理用 `pnpm`（依 `node-ts` skill）。
- 程式碼內不寫註解；顯示訊息用英文。
- 「進度」以**媒體時間的已涵蓋秒數**（covered seconds，區間聯集）計算，不是牆上時鐘。
- 「完成」= 收到 `ended` 事件 **且** 涵蓋率 ≥ `completionRatio`（預設 0.95）。
- 一個 URL key 對應一份字幕；一頁多個媒體時，MVP 只追蹤「主要媒體」（正在播放且面積最大的那個）。

---

## 6. Proposed architecture（架構）

### 6.1 元件與職責

```
┌──────────────── Web page (untrusted) ────────────────┐
│  <video>/<audio>                                      │
│  ┌───────── content script (isolated world) ───────┐  │
│  │ MediaDetector      找主要媒體、監聽 play/pause/   │  │
│  │                    seeked/ratechange/ended        │  │
│  │ AudioTap           captureStream → MediaRecorder  │  │
│  │                    切成 N 秒 chunk（含媒體時間）  │  │
│  │ CaptionOverlay     Shadow DOM tooltip，只用       │  │
│  │                    textContent 渲染               │  │
│  │ PlaybackController 依 currentTime 查 cue          │  │
│  │ PageUrlWatcher     SPA 換頁偵測 → 重新判斷        │  │
│  └───────────────┬────────────────────────────────┘  │
└──────────────────┼────────────────────────────────────┘
                   │ chrome.runtime Port（typed messages）
┌──────────────────▼──── background service worker ────┐
│ MessageGuard         驗 sender.id、schema               │
│ SiteAccessPolicy     allowlist / denylist 判斷          │
│ CaptionSessionCoordinator  每個 urlKey 一把 lease       │
│ TranscriptionQueue   重試、backoff、backlog 上限        │
│ AsrProvider (interface)                                 │
│   └ OpenAiCompatibleHttpProvider                        │
│ SubtitleStore (interface)                               │
│   ├ ServerSubtitleStore                                 │
│   └ IndexedDbSubtitleStore                              │
│ 持有 server token；content script 不碰 token            │
└──────────────────┬─────────────────────────────────────┘
                   │ HTTP（127.0.0.1，token header）
┌──────────────────▼──── caption-server（若選 ASR-A）────┐
│ POST /v1/audio/transcriptions  → mlx-whisper / Qwen3    │
│ GET  /subtitles/{urlKey}       → srt + meta             │
│ PUT  /subtitles/{urlKey}       → CAS 覆蓋               │
│ DELETE /subtitles/{urlKey}                              │
└────────────────────────────────────────────────────────┘

shared/（純函式，content / background / server 測試共用規格）
  SrtCodec · UrlCanonicalizer · CoverageTracker · OverwritePolicy
  CueTimeline · CueMerger · messages · config schema
```

### 6.2 為什麼這樣切（避免日後重複實作）

- **網路請求與 credential 只在 background**：content script 的 `fetch` 受頁面 CORS/CSP 影響，也不該拿 token。日後加 side panel、popup 匯出、其他 UI 入口，全部走同一個 background 服務，不各自打 server。
- **`AsrProvider` / `SubtitleStore` 是 Strategy**：換雲端、換瀏覽器內模型、加「直接抓 YouTube 既有字幕」的 provider，都不改流程碼。
- **`UrlCanonicalizer` 用規則表（site rules）**：YouTube、Bilibili 各自一條規則，不在流程裡寫 `if (host === ...)`。
- **`OverwritePolicy` 是單一純函式**：extension 和 server 都用同一份判斷規則（server 端用相同測試案例表驗證），避免兩邊規則漂移。

### 6.3 專案目錄

```
caption/
  package.json
  tsconfig.json
  build.mjs                 esbuild entries → dist/
  static/manifest.json
  src/
    shared/
      srt/srt-codec.ts
      url/url-canonicalizer.ts
      url/site-rules.ts
      progress/coverage-tracker.ts
      progress/overwrite-policy.ts
      cue/cue-timeline.ts
      cue/cue-merger.ts
      messages.ts
      config.ts
    content/
      index.ts
      media-detector.ts
      audio-tap.ts
      caption-overlay.ts
      playback-controller.ts
      page-url-watcher.ts
    background/
      index.ts
      message-guard.ts
      site-access-policy.ts
      caption-session-coordinator.ts
      transcription-queue.ts
      providers/asr-provider.ts
      providers/openai-compatible-http-provider.ts
      stores/subtitle-store.ts
      stores/server-subtitle-store.ts
      stores/indexed-db-subtitle-store.ts
    popup/   (狀態、此網站啟用/停用、匯出 .srt、刪除紀錄)
    options/ (endpoint、token、語言、allowlist)
  test/
    unit/
    e2e/
  server/   （若選 ASR-A：Python + uv，見 Phase 4）
```

---

## 7. Main flow（主流程）

### 7.1 頁面載入 / SPA 換頁

1. `PageUrlWatcher` 取得目前 URL（含 SPA 換頁事件）。
2. 送 `ResolvePage{url}` 給 background。
3. background：
   1. `SiteAccessPolicy` 不允許 → 回 `Disabled`，content 不做事。
   2. `UrlCanonicalizer` → `canonicalUrl` → `urlKey = sha256(canonicalUrl)`（hex）。
   3. `SubtitleStore.get(urlKey)`：
      - `complete` → 回 `Playback{cues}`。
      - `partial` → 回 `Recognize{existingCoveredSeconds, existingCues}`。
      - 無 → 回 `Recognize{existingCoveredSeconds: 0}`。
4. content 依結果進入播放模式或辨識模式。

### 7.1.1 YouTube 既有字幕（Q7 已決定）

1. 在 `SubtitleStore.get` 之前，YouTube 頁面先走 `ExistingCaptionProvider`：讀播放器的 caption track 清單（`ytInitialPlayerResponse.captions`，content script 要從頁面主世界讀，走 `chrome.scripting.executeScript({world:'MAIN'})`）。
2. 有中文字幕 → 轉繁體後直接進播放模式。
3. 只有英文字幕 → 交給 background 翻譯成繁中，產出雙語 SRT（英文原文一行 + 繁中一行），存進 store 標 `source=youtube-captions`，之後再開同一部影片直接讀 store。
4. 都沒有 → 走一般 ASR 辨識模式。
5. 優先順序：人工字幕 > 自動字幕（自動字幕可設定是否採用，預設採用）。

### 7.2 播放模式

1. `PlaybackController` 監聽 `timeupdate`（或 `requestVideoFrameCallback`）。
2. `CueTimeline.find(currentTime)`（二分搜尋）→ `CaptionOverlay.show(text)`。
3. 不建立 AudioTap，不呼叫 ASR。

### 7.3 辨識模式

1. `CaptionSessionCoordinator.acquire(urlKey, tabId)`：同一 urlKey 只有一個分頁能寫；拿不到 lease 的分頁只做顯示（從持有者廣播的 cue 顯示）。
2. 媒體 `play` 時 `AudioTap` 開始；每個 chunk 帶 `{sessionId, seq, mediaStart, mediaEnd, playbackRate}`。
   - `pause` → 結束目前 chunk。
   - `seeked` → 結束目前 chunk，下一個 chunk 從新的 `currentTime` 開始。
   - `playbackRate ≠ 1` → 依設定 `recognizeOnlyAtNormalRate` 決定略過或照樣辨識（時間用媒體時間換算）。
3. chunk 經 Port 送 background → `TranscriptionQueue` → `AsrProvider.transcribe()` → 得到相對時間的 segments → 加上 `mediaStart` 位移 → `CueMerger` 去除 chunk 重疊區的重複句。
4. 新 cue 回 content → `CaptionOverlay` 即時顯示。
5. `CoverageTracker` 累加已涵蓋區間（聯集，拖回去重聽不會重算）。
6. 每 `flushIntervalSeconds` 或每 N 個 cue，background 判斷 `OverwritePolicy`（§7.4）決定要不要寫入。
7. `ended` 且涵蓋率 ≥ `completionRatio` → 寫入 `status=complete`。
8. 分頁關閉 / 換頁 → 最後 flush 一次（遵守 OverwritePolicy）→ 釋放 lease。

### 7.4 覆蓋規則（OverwritePolicy）

輸入：`existing`（store 內紀錄，可能為空）、`candidate`（本 session 狀態）。

| existing | candidate | 結果 |
|---|---|---|
| 無 | 任意（coveredSeconds > 0） | 寫入 |
| `complete` | 任意 | **不寫**（照理不會進辨識模式；防呆） |
| `partial`，covered = X | covered ≤ X | 不寫，保留舊檔 |
| `partial`，covered = X | covered > X | **整份覆蓋** |
| `partial` | `complete` | 寫入（完成一定勝出） |
| 任意 | 來自同 session、seq 比已寫的舊 | 不寫（重試/亂序保護） |

- 寫入一律帶 `expectedRevision`（CAS）。store 內 revision 不符 → 重讀後重新判斷，避免兩個分頁/兩次重試互相蓋掉。
- 「超過」是嚴格大於。
- 在新 session 超過舊檔之前，overlay 顯示「本次即時結果」；本次還沒涵蓋到的時間點，改顯示舊 partial 的 cue（Q2 已決定）。

### 7.5 SRT 與 meta 格式

`<urlKey>.srt`：標準 SRT（`index`、`HH:MM:SS,mmm --> HH:MM:SS,mmm`、文字、空行），時間 = 媒體時間。

`<urlKey>.meta.json`：

```json
{
  "schemaVersion": 1,
  "urlKey": "<sha256 hex>",
  "canonicalUrl": "https://www.youtube.com/watch?v=xxxx",
  "status": "partial",
  "coveredSeconds": 312.4,
  "coveredIntervals": [[0, 180.2], [240, 372.2]],
  "mediaDuration": 1260.5,
  "language": "zh",
  "engine": "mlx-whisper:large-v3-turbo",
  "revision": 7,
  "sessionId": "uuid",
  "lastSeq": 52,
  "updatedAt": "2026-11-19T10:00:00Z"
}
```

---

## 8. Configuration design（設定）

存在 `chrome.storage.local`，啟動時經 `config.ts` 的 schema 驗證 + 預設值合併；錯誤設定直接拒絕並在 options page 顯示。

| key | 預設（dev） | 說明 | production 是否必須明確指定 |
|---|---|---|---|
| `asr.provider` | `openai-compatible-http` | provider 名稱（allowlist 內才可用） | 是 |
| `asr.endpoint` | `http://127.0.0.1:8765` | ASR server | 是；非 localhost 必須 `https` |
| `asr.token` | 首次啟動 server 時產生，貼到 options | 只存在 background 可讀的 storage | 是（每人一把，不共用） |
| `asr.language` | `auto` | `auto` / `zh` / `en` … | 否 |
| `asr.model` | `JacobLinCool/TEA-ASR-1.1-mini` | 傳給 server；可選 `JacobLinCool/TEA-ASR-1.1`、`MarkChen1214/Breeze-ASR-25-MLX-5bit` | 否 |
| `store.backend` | `server` | `server` / `indexeddb` | 是 |
| `chunk.seconds` | `6` | 每段長度（延遲 vs 準確度取捨） | 否 |
| `chunk.overlapSeconds` | `1` | 重疊秒數，配合 CueMerger | 否 |
| `flush.intervalSeconds` | `10` | 寫入頻率 | 否 |
| `completion.ratio` | `0.95` | 完成判定 | 否 |
| `queue.maxBacklogChunks` | `5` | 積壓超過就丟最舊的（即時性優先） | 否 |
| `queue.maxRetries` | `3` | 每個 chunk 重試次數 | 否 |
| `sites.mode` | `all-except-denylist` | 已決定所有網站；`allowlist` 保留為選項 | 是 |
| `sites.allowlist` | `[]` | `sites.mode=allowlist` 時才用 | 否 |
| `sites.denylist` | 常見銀行、會議網域 | 永不擷取 | 是 |
| `url.stripParams` | `utm_*`, `fbclid`, `gclid`, `si`, `t`, … | 正規化時移除 | 否 |
| `url.siteRules` | YouTube（只留 `v`）、youtu.be 轉址 | 每站規則 | 否 |
| `overlay.*` | 字級、位置、最多行數、透明度 | 顯示 | 否 |
| `features.autoStart` | `true` | 允許網站上自動開始 | 否 |
| `features.tabCaptureFallback` | `true` | 會議類頁面（WebRTC）點擊 extension 圖示啟用 tabCapture（Q5 已決定要支援） | 否 |
| `youtube.useExistingCaptions` | `true` | YouTube 有字幕就直接用（Q7） | 否 |
| `translate.model` | `mlx-community/HY-MT1.5-1.8B-8bit` | 翻譯模型（server 內用 mlx-lm 跑，不需 Ollama） | 否 |
| `translate.keepTerms` | `["transformer","token","attention", …]` | 術語詞表，翻譯時保留英文 | 否 |
| `translate.bilingualEnglish` | `true` | 英文句輸出「英文原文 + 繁中翻譯」兩行（Q9） | 否 |
| `playback.recognizeOnlyAtNormalRate` | `true` | 非 1x 速度時暫停辨識 | 否 |

Server 端設定（env）：`CAPTION_DATA_DIR`、`CAPTION_BIND`（預設 `127.0.0.1`）、`CAPTION_PORT`、`CAPTION_TOKEN`、`CAPTION_ALLOWED_ORIGIN`（`chrome-extension://<id>`）、`CAPTION_MAX_CHUNK_BYTES`、`CAPTION_ASR_BACKEND`。

---

## 9. Permissions and trust boundaries（權限與信任邊界）

### 9.1 信任層級

| 層 | 可信度 | 可以做 | 不可以做 |
|---|---|---|---|
| 網頁本身 | 不可信 | 播放媒體 | 碰 extension 訊息、token |
| content script | 半可信（DOM 可被頁面竄改） | 擷取音訊、顯示 overlay | 持有 token、直接打 server、直接寫 store |
| background | 可信 | 打 ASR、讀寫 store、判斷覆蓋 | 接受未驗證的訊息 |
| caption-server | 可信 | 辨識、存檔 | 主動抓任何 URL（不做 SSRF 面） |

### 9.2 Manifest 權限（最小化）

- `permissions`: `storage`, `offscreen`（fallback 才需要）, `tabCapture`（fallback 才需要）。
- `host_permissions`: 只放 ASR server（`http://127.0.0.1:8765/*`）。
- content script 的網站範圍：用 `optional_host_permissions` + `chrome.scripting.registerContentScripts`，**依 allowlist 動態註冊**，不預設 `<all_urls>`。
- 不要 `downloads`，除非選了儲存方案 B（匯出用）。

### 9.3 寫入（mutation）保護

- 寫/覆蓋/刪除字幕是受保護操作：
  - 只有 background 能呼叫 `SubtitleStore.put/delete`；content script 只能送「chunk」與「狀態」，不能送「請覆蓋」。
  - 覆蓋一律經過 `OverwritePolicy` + CAS revision；server 端**再驗一次同樣規則**（不信任 client 的判斷），防止繞過 extension 直接打 API 把好字幕蓋掉。
  - 刪除只從 options/popup（使用者手勢）觸發。
- Server 不接受任意路徑：`urlKey` 必須符合 `^[0-9a-f]{64}$`，檔案路徑由 server 組，杜絕 path traversal。
- Server 寫入 audit log（時間、urlKey、舊/新 revision、舊/新 coveredSeconds、sessionId）。

---

## 10. Security risks and mitigations（安全風險）

| 風險 | 說明 | 對策 |
|---|---|---|
| 隱私外洩 | 自動擷取所有網頁聲音（銀行影片、公司會議、私人通話） | 預設 allowlist；denylist 永遠優先；擷取中 overlay 顯示紅點指示；ASR 預設本機 |
| 本機 server 被其他網頁打（CSRF / DNS rebinding） | 任何網頁都能 `fetch('http://127.0.0.1:8765')` | 必須 token header；檢查 `Origin` = extension ID；檢查 `Host` 為 `127.0.0.1`/`localhost`；bind `127.0.0.1` |
| Token 外洩 | 放在 content script 會被頁面讀到 | token 只在 background；content 拿不到 |
| 偽造訊息 | 其他 extension 或頁面送訊息給 background | `MessageGuard` 驗 `sender.id === chrome.runtime.id`、`sender.tab` 存在、schema 驗證；不開 `externally_connectable` |
| XSS | SRT 內容（ASR 輸出或被竄改的檔案）含 HTML | overlay 只用 `textContent`；Shadow DOM 隔離 |
| Path traversal | urlKey 組檔名 | 嚴格 hex 驗證 |
| URL 含敏感參數 | 例如 `?token=` 被存進 meta | 正規化時只保留 site rule 指定參數；未知網站預設移除全部 query（可設定）；檔名用 hash |
| 資源濫用 | 長時間播放 → 大量 ASR 呼叫 | backlog 上限、chunk 大小上限、server rate limit |
| 惡意頁面假媒體 | 頁面塞假 `<video>` 造假字幕污染 key | key 綁 canonical URL；影響只限該 URL；可在 popup 刪除 |

---

## 11. Scalability and future maintenance（擴充與維護）

1. **其他 UI 入口**（side panel 字幕列表、搜尋、匯出）：全部呼叫 background 的同一組 message API，不各自打 server。
2. **其他 ASR**（雲端、瀏覽器內、YouTube 既有字幕）：新增 `AsrProvider` 實作 + provider allowlist，不動流程。
3. **翻譯**：`Translator` 是獨立介面（server 內 mlx-lm 或外部 OpenAI 相容 endpoint），不混進 ASR provider；YouTube 英文字幕與 ASR 英文句共用同一個 Translator。
4. **團隊共用 / 多副本 server**：`SubtitleStore` 換成 DB（revision 欄位做 CAS），key 加 user/team namespace；ASR 改 queue + worker。
5. **冪等與重試**：chunk id = `sessionId:seq`；server 對重複 chunk 回相同結果；寫入 CAS。
6. **Metrics**：每 chunk 延遲、失敗率、積壓數，background 記在 `chrome.storage.session`，popup 顯示；server 輸出 log。
7. **Rollback**：meta 有 `schemaVersion`；server 覆蓋前把舊檔留成 `<urlKey>.srt.prev`（只留一份）。
8. **Timeout**：ASR 請求逾時（預設 30s）視為失敗進重試；超過 maxRetries 丟棄並在 overlay 標示缺段。

---

## 12. Implementation phases（實作階段）

每一階段結束都要有對應測試綠燈才進下一階段。

### Phase 0：可行性 probe（先證偽最大風險）

| Probe | 驗收 |
|---|---|
| `captureStream()` 在 YouTube、同源 mp4、跨源無 CORS mp4 是否拿得到聲音 | 記錄每種的結果（有聲/靜音/例外） |
| MediaRecorder「每 N 秒 stop/start」產生可獨立解碼的 webm/opus chunk；量 chunk 之間的空隙 | 空隙 < 50ms，或改用雙 recorder 交錯 |
| （備案）AudioWorklet 在 content script 是否被頁面 CSP 擋 | 記錄結果 |
| 本機 ASR 處理 6 秒 chunk 的延遲（mlx-whisper large-v3-turbo vs Qwen3 ASR） | 延遲 < chunk 長度，才稱得上即時 |

Probe 追加：

| Probe | 驗收 |
|---|---|
| C. ASR 模型本機實測（TEA-ASR-1.1、TEA-ASR-1.1-mini、Breeze-ASR-25 MLX Q5）：NTUML2021 40 段台灣課堂中英混合 + 6 段英文 | MER、每段延遲 p50/p95、RTFx、輸出是否為繁體、英文句是否保持英文 |
| D. 英翻繁模型（本機 OpenAI 相容 endpoint） | 6 句英文翻譯品質與延遲 |

#### Phase 0 結果（2026-11-19，Chrome 157 headless，M4 Max）

| Probe | 結果 |
|---|---|
| A. `captureStream()` 同源 wav | ✅ 有聲（peak 0.79） |
| A. 跨源、無 CORS | ❌ `SecurityError: Cannot capture from element with cross-origin data` → 這類頁面要走 tabCapture（需點擊） |
| A. 跨源、有 CORS + `crossOrigin` | ✅ 有聲（但 extension 不能替頁面改 `crossOrigin`，實務上視同上一列） |
| A. YouTube（`video` 元素，MSE） | ✅ 有聲（peak 0.55） |
| B. MediaRecorder 每 2 秒 stop/start | ✅ 每段 webm/opus 都能獨立 decode；每段 decode 長度 1.98s vs 媒體時間約 2.00s → 每次重啟約掉 20–45ms，用 `chunk.overlapSeconds` 補 |
| YouTube 字幕清單 | ✅ 主世界可讀 `captionTracks`（測試影片有 zh-TW 人工、en 人工、en 自動）；SPA 換片時 `ytInitialPlayerResponse` 會過期，要改讀播放器的 player response |

#### Phase 0 probe C/D 結果（2026-11-19，M4 Max，MPS / MLX）

測試集：NTUML2021 test 前 40 段（128.5s，台灣課堂中英混合；注意 TEA 與 Breeze 訓練資料都含 NTUML2021，數字可能偏樂觀）＋ macOS `say` 英文 6 句（22.1s，合成音）。MER 計算：兩邊 OpenCC t2s、小寫、去標點；中文逐字、英文逐詞。延遲 = 單段（2.4–4.4s）處理時間，模型已 warm-up。

| 模型 | 設定 | ML2021 MER | 延遲 p50 / p95 | RTFx | 繁體比例 | 英文 6 句 MER |
|---|---|---|---|---|---|---|
| TEA-ASR-1.1-mini（torch MPS bf16） | `language="Chinese"` | **4.98%** | 0.25s / 0.36s | 12.0x | 0.97（會漏「来」「这」） | 2.94%（英文保持英文） |
| TEA-ASR-1.1-mini | 自動語言 | 5.72% | 0.21s / 0.25s | 15.1x | 0.97 | 2.94% |
| Breeze-ASR-25 MLX Q5 | 自動語言 | 6.09% | 0.24s / 0.28s | 13.3x | 1.00 | 0.00% |
| TEA-ASR-1.1（2B，torch MPS bf16） | `language="Chinese"` | 6.46% | 0.36s / 0.45s | 9.0x | 1.00 | 2.94% |

結論：在這 40 段（542 個 token）上 2B **沒有比 mini 好**，差距約 8 個 token，部分是評分 artifact（把 `nat` 唸成 `n a t`），樣本太小不足以說 2B 較差，但也沒有證據值得多花 1.4 倍延遲。→ **預設 `asr.model = JacobLinCool/TEA-ASR-1.1-mini` + `language="Chinese"` + OpenCC `s2tw` 補字**；TEA-ASR-1.1 與 Breeze-ASR-25 MLX Q5 保留為設定選項。之後若有更多真實網頁語音，再用同一支 `bench.py` 重比。

翻譯（probe D）：`mlx-community/HY-MT1.5-1.8B-8bit`，6 句英文 p50 0.27s、最慢 0.47s，直接輸出繁體。
- 問題 1：技術詞被硬翻（transformer→變換器、token→代碼）。用 HY-MT 的術語提示（`参考下面的翻译：\n<term> 翻译成 <term>`）帶入詞表後，transformer / token / attention 都保留英文 → 設定 `translate.keepTerms`（詞表）。
- 問題 2：OpenCC `s2twp`（含詞彙轉換）會把「代碼」改成「程式碼」造成誤譯 → 翻譯輸出只用 `s2tw`（字形轉換），ASR 輸出同樣用 `s2tw` 補漏網的簡體字。
- 問題 3：小模型偶爾加料（「提交給老師」）。因為 Q9 決定雙語顯示（英文原文 + 繁中），誤譯可由原文對照；需要更好品質時可換 7B（設定 `translate.model`）。


### Phase 1：專案骨架 + shared 純函式

- `caption/` 建立：`pnpm init`、TypeScript、esbuild `build.mjs`、vitest、`static/manifest.json`。
- 實作並測試：`SrtCodec`、`UrlCanonicalizer`、`CoverageTracker`、`OverwritePolicy`、`CueTimeline`、`CueMerger`、`config` schema。

### Phase 2：播放模式

- background：`MessageGuard`、`SiteAccessPolicy`、`SubtitleStore`（先用 `IndexedDbSubtitleStore` + fixture）。
- content：`MediaDetector`、`PlaybackController`、`CaptionOverlay`（Shadow DOM、可拖曳、全螢幕時掛到 `document.fullscreenElement`）。

### Phase 3：辨識模式

- `AudioTap`、`TranscriptionQueue`、`OpenAiCompatibleHttpProvider`、`CaptionSessionCoordinator`、flush + OverwritePolicy。

### Phase 4：caption-server（若選 ASR-A / 儲存-A）

- `caption/server/`，Python + uv；OpenAI 相容 transcriptions endpoint + subtitles CRUD；token/Origin/Host 檢查；CAS；audit log。
- 與 extension 共用 OverwritePolicy 測試案例表（JSON fixture，兩邊都讀）。

### Phase 5：穩健性

- SPA 換頁、seek/pause/rate、多分頁 lease、server 斷線（降級成只播放）、tabCapture fallback（需點擊）。

### Phase 6：popup / options

- 此網站啟用/停用、目前模式與進度、匯出 `.srt`、刪除紀錄、endpoint/token/語言設定。

---

## 13. Test plan（測試）

| 層 | 工具 | 內容 |
|---|---|---|
| 單元 | vitest | SrtCodec round-trip（含 >1 小時、毫秒進位、多行文字、CRLF）；UrlCanonicalizer 規則表；CoverageTracker 區間聯集（重疊、拖回、亂序）；OverwritePolicy 全部表格案例；CueTimeline 邊界；CueMerger 重疊去重；config 驗證 |
| 整合 | vitest + fake provider/store | background 收 chunk → provider → merge → flush；CAS 衝突重讀；重試與 backlog 丟棄；lease 搶占 |
| Server | pytest | urlKey 驗證、token/Origin/Host、CAS、OverwritePolicy 共用 fixture、chunk 大小限制 |
| E2E | Playwright（載入 unpacked extension 的 Chromium） | 本機測試頁播放一段已知語音的 mp4（用 macOS `say` + ffmpeg 產生）：①第一次 → overlay 出現預期文字、store 有 partial/complete；②重新整理 → 播放模式、ASR 呼叫次數 = 0；③先跑 30% 中斷，再跑到 40% → 被覆蓋；只跑到 20% → 不覆蓋 |
| 手動 | — | YouTube 真實影片、全螢幕、SPA 換片、兩個分頁同一支影片 |

驗收前依「閉包 ∩ 改動」挑關卡：只改 shared 純函式時跑單元即可，不必跑 E2E。

---

## 14. Local/dev 與 production 差異

| 項目 | local/dev | production（若之後要給團隊用） |
|---|---|---|
| Extension 安裝 | unpacked（`caption/dist`） | Chrome Web Store / 企業政策安裝，固定 extension ID |
| ASR endpoint | `http://127.0.0.1:8765` | `https://` 內部服務；host_permissions 改成該網域 |
| Token | server 啟動時產生，手動貼 options | 每人一把，由 SSO 換發；可撤銷 |
| CORS / Origin | dev extension ID | 正式 extension ID（env 指定，不寫死） |
| 儲存 | 本機資料夾 | 持久化 volume 或 DB；備份 |
| 網站範圍 | allowlist 手動加 | 預設 allowlist 由 policy 下發 |
| Log | verbose | 不記錄字幕內容，只記 metadata |
| 隱私揭露 | — | Web Store 需要隱私說明（音訊處理） |

可暫用預設值：chunk、flush、overlay、completion ratio。
production 必須明確指定：endpoint、token、allowed origin、data dir、site policy。

---

## 15. Open questions

Q1–Q9 已全部回覆，決定見 §0。仍待確認：

1. 「所有網站自動擷取」代表銀行、私人影片的聲音也會送到本機 caption-server（不出本機）。若之後 server 改成遠端，需重新評估隱私；目前保留 denylist 設定（預設空）。
2. ~~翻譯模型~~：已定案 HY-MT1.5-1.8B-8bit（server 內 mlx-lm，不需 Ollama），見 probe D。

---

## 16. 實作紀錄與偏離原計畫之處（2026-11-19）

### 已完成

| Phase | 內容 | 驗證 |
|---|---|---|
| 1 | `caption/`：pnpm + TypeScript 7 + vitest + zod。`src/shared/`：`srt/srt-codec.ts`、`url/url-canonicalizer.ts` + `url/default-site-rules.ts`、`progress/coverage.ts`、`progress/overwrite-policy.ts`、`cue/cue.ts`、`cue/cue-timeline.ts`、`cue/cue-merger.ts`、`config.ts` | `pnpm test` 68 passed；`pnpm typecheck` 無錯 |
| 4 | `caption/server/`（uv，Python 3.12，FastAPI）：`settings`、`audio_decoder`（PyAV → 16k mono）、`text_normalizer`（OpenCC `s2tw`、語言判斷、術語偵測）、`engines`（`QwenAsrRecognizer`、`HyMtTranslator`）、`caption_pipeline`、`overwrite_policy`、`subtitle_store`（原子寫入、`.srt.prev`、per-key lock、CAS、audit）、`security`（token / Origin / Host）、`app` | `uv run pytest -q` 54 passed；真模型 smoke：中文 3 段延遲 0.32–0.57s、文字與參考幾乎一致；英文 3 段 0.5–0.7s 雙語輸出；PUT/GET、壞 Origin 403、無 token 401 |

### 偏離原計畫（以這裡為準）

1. **切段方式**：原計畫「固定 6 秒 + 1 秒重疊 + 文字去重」改為「**靜音切段**」：最短 `chunk.minSeconds`(2)、最長 `chunk.maxSeconds`(8)、靜音 `chunk.silenceSeconds`(0.3) 即切。理由：Qwen3-ASR/TEA 只回文字、**不回時間戳**（要時間戳需額外的 forced aligner 模型），重疊段的文字去重不可靠；在靜音處切，MediaRecorder 重啟掉的 20–45ms 也落在靜音裡。每段 = 一個 cue（`[mediaStart, mediaEnd]`）。`CueMerger` 因此只做「時間區間覆蓋取代」。
2. **Server API**：不做 OpenAI 相容 `/v1/audio/transcriptions`，改為 `/v1/captions/transcribe`（ASR → `s2tw` → 語言判斷 → 英文句翻譯 → `display`）一次完成，少一次往返；另有 `/v1/translations`（YouTube 英文字幕用）。詳見 `caption/server/README.md`。
3. **SRT 由 extension 產生**，server 只存字串、不解析 SRT（避免 TS / Python 各寫一份 codec）。覆蓋規則兩邊各一份，但共用 `caption/test/fixtures/overwrite-policy-cases.json` 測試案例防漂移。
4. **模型設定在 server 端 env**（`CAPTION_ASR_MODEL`、`CAPTION_TRANSLATE_MODEL`…），extension 設定不再有 `asr.*` / `translate.*`。
5. **URL 正規化**：未知網站**保留**非追蹤用的 query（原計畫是全部移除）。理由：很多影片站用 query 區分影片（`?id=42`），全移除會讓不同影片共用同一份字幕。
6. **Server 不送 CORS header**：extension background 靠 `host_permissions` 呼叫，一般網頁讀不到回應。
7. **MLX 執行緒**：mlx-lm 在 import 時建立 GPU stream，綁定該執行緒；翻譯模型的 import、load、generate 全部固定在單一 `mlx` 執行緒。
8. **術語偵測**含複數（`transformers`）並回傳原文形式，否則 `transformers` 會被翻成「變形金剛」（smoke 實際發生過）。

### Phase 2/3/5/6（extension 本體，2026-11-19）

| 項目 | 內容 |
|---|---|
| build | `caption/build.mjs`（esbuild，minify）、`static/manifest.json`（MV3，`storage`/`tabCapture`/`offscreen`/`activeTab`，host 只放 127.0.0.1/localhost，遠端 https 走 `optional_host_permissions`） |
| background | `caption-server-client`（唯一持有 token）、`page-resolver`、`config-repository`、`tab-status-registry`（badge：SRT/REC/CC/TAB）、`tab-capture-coordinator`、`index`（依 sender 分流：offscreen → extension 頁面 → content） |
| content | `page-caption-controller`（第一次 `playing` 才問 server）、`media-asr-source` + `shared/audio/audio-tap`（captureStream + MediaRecorder + 靜音切段）、`transcription-queue`（backlog/retry）、`youtube-caption-reader`、`tab-capture-receiver`、`caption-overlay`（closed Shadow DOM、✕ 關閉、全螢幕跟隨）、`subtitle-saver` |
| offscreen / popup / options | tab 音訊擷取並回放；popup：狀態、擷取 tab 音訊、刪除本頁字幕、設定；options：endpoint、token、網站清單、字級、開關 |
| 驗證 | `pnpm check`：typecheck + 105 unit tests + build；`pnpm e2e`（真 server + Playwright Chromium）：14.7 秒中英混合語音 → SRT `complete`、涵蓋 14.68 秒、英文句雙語、第二次造訪 0 次 ASR 呼叫、options「Test connection」成功；截圖確認辨識中與播放時 overlay 有字幕與 ✕ |

### 再次偏離原計畫

9. **YouTube 字幕（Q7）改為讀播放器畫面上顯示的字幕 DOM**，邊播邊存成 SRT（英文行經 `/v1/translations` 變雙語）。原因：(a) 直接抓 caption track（`/api/timedtext`）在自動化瀏覽器回空內容（需 PO token）；(b) 改呼叫 YouTube 內部 `get_transcript` API 的實測被 auto mode 以 Unauthorized Access 擋下，不繞過。代價：不是一開頁就有完整字幕，要播放過才會存滿；跟 ASR 共用同一套 session / 覆蓋規則。
10. **content / offscreen 不用 zod**（zod v4 會帶進 264KB locales），改用 `shared/message-guards.ts` 手寫檢查；content.js 從 478KB 降到 21KB。
11. **辨識中的顯示**：即時結果一定比播放時間晚一段，所以顯示「最近一句」並保留 6 秒（`findRecentCue`），否則字幕永遠不會出現（e2e 截圖抓到的 bug）。
12. 關閉 ✕ = 本頁隱藏字幕並停止辨識（先把已辨識部分存檔），換頁或重新整理後恢復。

### 使用者實測 bug 修正（2026-11-19 16:10）

- **現象**：YouTube `mAbdM_F0sQI` 沒有 tooltip；server 存了 `source=youtube-captions`、涵蓋 49.5 秒、SRT 0 bytes；Capture tab audio 沒有任何寫入。
- **根因 1**：該影片沒有字幕（`captionTracks` = 0），但 CC 按鈕仍顯示、沒有 `aria-disabled`，只有標籤寫「未提供字幕」；舊判斷把它當成「有字幕」，走字幕路徑、讀不到字、只累積時間。
- **根因 2**：tab capture 開了新 session，依覆蓋規則必須超過舊 session 的 49.5 秒才能寫入。
- **修正**：(a) 點 CC 後看 `aria-pressed` 是否變 `true` 判斷有沒有字幕（實測：無字幕影片維持 `false`、有字幕影片為 `true`，與語系無關）；(b) 字幕路徑看到第一行字才開始累計涵蓋，播放 20 秒都沒字就自動改走 ASR 並還原 CC 按鈕；(c) 同一頁面只用一個 session，切換來源（YouTube 字幕 / ASR / tab capture）沿用同一個 sessionId。
- **驗證**：`pnpm e2e:youtube`（真 YouTube 無字幕影片）→ 直接走 ASR、SRT 有內容（「我叫許嵐，做了八年專業管家。」）、CC 按鈕維持原狀、截圖有 tooltip；`pnpm e2e` 仍通過；unit 107 passed。e2e server 改用 8766，避免和 `dev.sh` 的 8765 衝突。
- 使用者 server 上那筆空紀錄已用 DELETE API 刪除（audit 有記錄）。
- 已知：tooltip 預設位置會壓到影片內嵌（燒進畫面）的字幕。

### 使用者回饋：拖曳、同步、常駐框（2026-11-19 16:30）

- **拖曳**：tooltip 可拖曳，位置以「相對影片大小的比例」存在 config `overlay.offsetX/offsetY`（經 background `save-overlay-offset` 寫入，content 不碰 storage）；雙擊回到預設位置；點擊不會傳到 YouTube 播放器。
- **常駐框**：tooltip 框一直顯示（固定最小寬高），只有裡面的文字換來換去。
- **同步（interim）**：live 辨識不可能零延遲。方案比較後選「暫時字幕」：錄音中每 1 秒（`chunk.interimSeconds`）把這段目前錄到的音訊送去辨識（`translate=false`，不翻譯），先顯示；整段結束後換成正式結果並存檔。queue 規則：正式結果優先、暫時請求只留最新一個、過期的暫時結果丟棄。實測：句子第 1 秒開始講，第 3 秒畫面已顯示「接下來我們要」（原本要等到第 6.2 秒）。代價：ASR 呼叫數變多（14.7 秒片段 3 → 14 次）。
- 未採用：只縮短切段（仍 2–4 秒延遲、切斷句子）；server 預先下載 YouTube 全片辨識（yt-dlp 有條款風險）。
- 驗證：unit 108、server 58、`pnpm e2e`（含 0.5 秒空框、3 秒暫時字幕、拖曳後位置存檔截圖）、`pnpm e2e:youtube` 全過。

### 兩行滾動字幕與 setup（2026-11-19 16:50）

- **字幕排版**：每句一行（依換行與 `。！？.!?` 切），目前這句在下、前一句在上，最多 2 行（`overlay.maxLines` 預設改 2）；英文雙語句本身就 2 行，單獨佔滿。直播辨識、YouTube 字幕、播放已存 SRT 三種模式共用 `shared/cue/caption-lines.ts`。
- **setup**：`caption/setup.sh` → `uv sync` + `uv run caption-server-setup`；要下載哪些模型由 Python 讀 server Settings 決定，模型放在 Hugging Face cache，不進 repo。
- 驗證：unit 122、server 61、`pnpm e2e` 通過（截圖：上一行前一句、下一行目前句）。

### 單行字幕 + 句子分段模型（2026-11-19 17:00）

- **單行**：只顯示目前這一句（`overlay.maxSentences` 預設 1，取代 `maxLines`）；英文句與其繁中翻譯視為同一句，一起顯示。
- **根因（使用者 SRT 證據）**：戲劇有背景音樂，RMS 靜音偵測不觸發，chunk 幾乎都在 8 秒上限被切（句子、詞被切斷）；部分 chunk 沒標點（「先生老太太把我的工資…聽見沒有我兒子」）。
- **方案比較**：A 標點恢復模型、B forced aligner、C 真正的 VAD（在網頁內跑 WebAssembly，可能被頁面 CSP 擋、改動大）。採 A + B，C 留待後續。
- **實作**：server `sentence_segmenter.py`：標點只「插回」原文（保留原文空白、全形/半形依子句語言）、句末 `。？！.?!` 切句、過長句在逗號處再切（`CAPTION_MAX_SENTENCE_UNITS`=24）、用 aligner 的逐字時間給每句起訖（數量對不上時按字數比例）；chunk 尾端不補假的「。/，」（保留「？！」）。`/v1/captions/transcribe` 回 `segments[]`；`final=false`（暫時字幕）不跑 aligner、不翻譯。前端每句記成一個 cue；tab capture 也逐句。
- **實測**：aligner 每段只多 0.06–0.16 秒；標點 5–12ms。YouTube 劇集同一段：原本 2.78→9.78 一整段 → 現在 3.85→6.09「三年前，林小姐，請我來林家。」、6.41→9.45「月薪三萬…」；對話會分開（「資料和接待的事兒你先準備。」/「好，好的，林小姐。」）。
- **仍存在**：8 秒硬切造成的跨段斷句（「…做滿」/「三年再給…」）要靠 C（VAD）才能根治；標點模型偶有誤判（「兩萬，七」）。自動化瀏覽器播 YouTube 約 38 秒後會被 YouTube 擋（顯示「發生錯誤」），長時間測試只能手動。
- 驗證：unit 123、server 76、`pnpm e2e`、`pnpm e2e:youtube` 通過。

### 待辦：VAD 切段（使用者 2026-11-19 指定先記下，之後再做）

- **問題**：背景音樂讓 RMS 靜音偵測失效，chunk 在 8 秒上限硬切，句子跨段被拆開（「…做滿」/「三年再給…」）。
- **作法**：用真正的語音活動偵測（VAD，例如 Silero VAD，約 2MB ONNX）判斷「有人聲 / 沒人聲」，在人聲停頓處切 chunk，取代 `chunk-cut-policy` 的 RMS 門檻。
- **放哪裡**：(a) content script 內用 onnxruntime-web（WebAssembly）跑，需確認頁面 CSP 是否擋 wasm、每頁載入成本；(b) 改成串流架構：前端每 0.5–1 秒送音訊、server 端跑 VAD + 切段 + ASR（需要 WebSocket 與 server 端 session 狀態）。先做 (a) 的可行性 probe（YouTube、一般網站各一）。
- **驗收**：同一部背景音樂戲劇，跨段斷句數明顯下降；chunk 長度分布不再集中在 8 秒。

### 覆蓋改為詢問使用者（2026-11-19 17:20，取代 Q3 的長短規則）

- **規則**（TS `overwrite-policy.ts`、Python `overwrite_policy.py`、共用 `test/fixtures/overwrite-policy-cases.json`）：沒有舊檔 → 直接存；同一 session → 新 seq 就存；**其他 session 的舊檔（不論長短、partial 或 complete）→ 必須帶 `confirmOverwrite: true` 才寫**，否則 server 回 `confirmation-required`。不再比較涵蓋秒數。
- **時機**：錄製後第一次存檔（約 10 秒，且至少已有一句字幕）時，字幕框下方出現「Replace the saved subtitles with this recording?」＋ Replace / Keep existing。Replace → 這次錄製之後都覆蓋；Keep existing → 這次不存，字幕照常顯示。若別的分頁在我們存檔後又寫入，server 回 `confirmation-required` → 再問一次。
- **安全**：按鈕在 closed Shadow DOM 內，且只接受 `event.isTrusted` 的點擊，網頁腳本無法代替使用者按「Replace」。
- **驗證**：unit 124、server 74、`pnpm e2e`（先錄一段 partial 後離開 → 再開同頁 → 出現詢問、回答前舊檔不變 → 按 Replace → 新 session 寫完 complete → 第三次開頁直接播放、0 次 ASR）。

### 詢問時機改到錄製之後、延遲量測（2026-11-19 18:30）

- **詢問時機**：不再在第一次存檔時（約 10 秒）就問；改成暫停超過 0.8 秒、影片播完、停止 tab capture、按 ✕ 時才問。回答前這次錄製都不存（只留在記憶體）；若沒回答就離開頁面，這次錄製不存，舊檔保持不變。
- **延遲量測工具**：`test/e2e/caption-latency.diagnose.mjs`（每 0.3 秒用 CDP 讀 closed Shadow DOM 裡的字幕文字，對照 SRT 每句的說話時間）。結論：暫時字幕約在說話後 1–1.5 秒出現；先前量到的 5–7 秒是因為整段最終辨識漏了一句（「還有一個月合同就到期了」），aligner 把後一句排到錯的時間，屬於量測假象。真正的體感延遲來自段落交界：上一段正式結果回來時又顯示舊句子、且正式結果排在下一段暫時字幕前面。
- **修正**：queue 改成暫時字幕優先（正式結果最多等 1.5 秒避免被丟掉）；舊句子保留時間 6 → 3 秒。新段落第一句提早約 0.4 秒出現。
- **限制**：以 1 秒為單位的暫時辨識，延遲下限約 1 秒；要再降需要 server 端串流辨識（Qwen3-ASR 有 streaming API），與 VAD 待辦一起考慮。
- **YouTube 自動化播放約 40 秒就出錯**：對照組（不裝 extension 的同一個自動化瀏覽器）也一樣，是 YouTube 擋自動化，不是 extension 造成。`test/e2e/youtube-long-play.diagnose.mjs` 可重現。

### 串流辨識 + VAD（使用者 2026-11-19 選「選項 2」）

- **Probe 結果**：(1) `qwen-asr` 的 `streaming_transcribe` 只支援 vLLM（NVIDIA），Mac 不能用；它本質是「每 2 秒把累積音訊重轉一次＋用前次輸出當 prompt」，所以改在 server 自己實作同一概念。(2) YouTube 上 `AudioContext({sampleRate:16000})` + ScriptProcessor 可取得 16kHz PCM；AudioWorklet 被 YouTube CSP 擋。(3) sherpa-onnx `VoiceActivityDetector` + Silero VAD（`istupakov/silero-vad-onnx`，2.3MB）在合成背景音樂下，句界誤差約 0.1 秒、0.25 秒停頓也抓得到，30 秒音訊 0.12 秒處理完。
- **架構**：content（ScriptProcessor 每 ~0.13 秒送 16kHz PCM + 媒體時間）→ background（Port → 一條 WebSocket，唯一持有 token）→ server `/v1/captions/stream`（每連線一個串流 session：VAD 切句；說話中每 0.5 秒重轉目前這句 → `partial`；VAD 判定句子結束 → aligner + 標點 + 翻譯 → `final`；定期回報 `covered`）。暫停 / 拖曳 / 廣告 / 變速 → client 送 `break`，server 收尾目前這句並重設 VAD。tab capture 走同一條 relay。
- **淘汰**：MediaRecorder 分段（`audio-tap`、`chunk-cut-policy`）、`transcription-queue`、`media-asr-source`、HTTP `/v1/captions/transcribe`、extension 設定的 `chunk` / `queue`（舊設定 key 改為忽略，不讓 token 因此遺失）。VAD / 間隔參數移到 server settings。
- **刪除流程（依「先量耦合」規則）**：先讓 controller 改走新路徑 → 量反向依賴（4 個前端檔只彼此互相引用、不是 build 進入點）→ 第 1 批刪檔 + typecheck/unit/build；第 2 批刪型別與設定；server 端先把 `TARGET_SAMPLE_RATE` 移到 `audio.py` 切斷耦合 → 再刪 HTTP 端點、`audio_decoder`、`av`、`python-multipart`。
- **結果**：YouTube 劇集前 34 秒，字幕在說話後 0.8–1.8 秒出現（中位數約 1.1 秒），先前 5–7 秒的離群值消失；字幕每 0.3–0.6 秒更新；VAD 依停頓切句，原本被漏掉的「還有一個月，合同就到期了。」與「兩萬七」都正確。
- **e2e 修正**：舊的「重播時 0 次 ASR」檢查改數 WebSocket 連線（原本改架構後會變成 0 = 0 的空檢查），並要求兩次錄製造訪都確實開了串流。
- **驗證**：unit 120、server 77、`pnpm e2e`、`pnpm e2e:youtube` 通過；量測工具 `test/e2e/caption-latency.diagnose.mjs`。

### 尚未自動驗證（需手動）

- YouTube 有字幕影片的 DOM 讀取路徑（自動化瀏覽器拿不到 YouTube 字幕）。
- 「Capture tab audio」（需要真人點 toolbar 給 `activeTab`；Meet 等 WebRTC 頁面）。
- 跨源無 CORS 影片顯示提示訊息的路徑。

### 已知限制

- `httpx` 測試用戶端有 Starlette 棄用警告；改用 `httpx2` 被 auto mode 擋下，暫不處理。
- 只接 Qwen3-ASR 系列（TEA mini / TEA 2B）；Breeze MLX 未接（需要它 repo 內的 `mlx_whisper_mixed.py`）。
