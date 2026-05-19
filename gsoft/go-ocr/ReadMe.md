# go-ocr

macOS 狀態列常駐應用程式：按熱鍵框選螢幕區域 → 本機 Ollama OCR → 結果寫入剪貼簿。

## 功能

- 狀態列 icon + 右鍵選單（`Setting` / `Quit`）
- 全域熱鍵 `Shift+Cmd+T` → 截圖框選 → 圖片進剪貼簿 → 呼叫 OCR → 文字蓋掉剪貼簿
- 全域熱鍵 `Ctrl+Cmd+T` → 把剪貼簿中的圖片送 OCR
- 全域熱鍵 `Shift+Cmd+R` → OCR + 自動翻譯（中→英 / 英→繁中）
- Setting 視窗可調整 endpoint / model / prompt / 熱鍵
- macOS 通知回報 OCR 與翻譯進度

## 前置條件

1. **Ollama** 在 `127.0.0.1:11434` 跑著
2. 一個有 vision capability 的模型給 OCR 用，例如：
   ```bash
   ollama pull qwen2.5vl:3b
   # 或
   ollama pull qwen3.6:35b-a3b-q4_K_M   # 本機已測過, 含 thinking mode
   ```
3. 一個翻譯小模型給 Translate 用（thinking 模型推理慢，獨立 model 加速翻譯）：
   ```bash
   ollama pull sun_leaf/HY-MT:1.8b      # 預設, 1.8B, 純文字翻譯
   ```

預設 endpoint 與兩個 model 可以在 `Setting` 視窗或 `~/Library/Application Support/go-ocr/config.json` 修改。

## 安裝

### 一次性：建立 stable code signing 憑證

```bash
./setup-cert.sh
```

建立 `go-ocr-codesign` 自簽憑證並匯入 login keychain。**這一步只要做一次**——之後 rebuild 都用同一張憑證簽，macOS TCC 權限（Screen Recording / Accessibility）不會因為重 build 而失效。

### 編譯並安裝到 `/Applications`

```bash
./install.sh
```

執行流程：
1. 跑 `build.sh`：編譯 Go binary、組 `.app` bundle、用 stable cert 簽
2. 停掉任何執行中的 instance
3. 複製到 `/Applications/go-ocr.app`
4. `open` 啟動

### 加入開機自動啟動

```bash
./install.sh --autostart
```

除了上面流程，再寫一份 LaunchAgent plist 到：

```
~/Library/LaunchAgents/com.flash.go-ocr.plist
```

並用 `launchctl load` 註冊，下次登入時 launchd 會自動啟動 go-ocr。

### 移除

```bash
./install.sh --uninstall
```

會停掉 process、unload LaunchAgent、刪掉 plist、刪掉 `/Applications/go-ocr.app`。

## 開發

```bash
./run.sh
```

從 `build/go-ocr.app/Contents/MacOS/go-ocr` 直接執行（不裝到 `/Applications`）。source 比 binary 新時自動 rebuild。

Log 永遠寫到 `~/Library/Logs/go-ocr.log`。

## 首次啟動需要的權限

第一次按熱鍵時 macOS 會分別跳出兩個權限對話框：

| 權限 | 用途 |
|------|------|
| **Accessibility** | `golang.design/x/hotkey` 註冊全域熱鍵 |
| **Screen Recording** | `screencapture -i -s` 抓螢幕 |

兩個都要允許。`System Settings → Privacy & Security` 裡找 `go-ocr` 開啟即可。授權之後 **下次 rebuild 不會再被要求**，因為用了 stable cert。

## 設定檔

```
~/Library/Application Support/go-ocr/config.json
```

```json
{
  "ocr_endpoint": "http://127.0.0.1:11434/v1/chat/completions",
  "ocr_model": "qwen3.6:35b-a3b-q4_K_M",
  "ocr_prompt": "Extract all visible text from this document image and return only the transcription in reading order using a markdown-first format. Use HTML only for tables. Use LaTeX only for formulas.",
  "translate_model": "sun_leaf/HY-MT:1.8b",
  "translate_prompt": "Translate the following text to {target_lang}. Output ONLY the translation, with no explanation, no quotes, no markdown, no language label.",
  "screenshot_hotkey": "shift+cmd+t",
  "clipboard_ocr_hotkey": "ctrl+cmd+t",
  "translate_hotkey": "shift+cmd+r"
}
```

`ocr_model` 需要 vision capability（看圖）；`translate_model` 是純文字翻譯小模型，可獨立替換以兼顧速度。

熱鍵字串格式：`modifier(s) + key`，例如 `shift+cmd+t`、`ctrl+cmd+t`、`opt+shift+f12`。
支援的 modifier：`cmd` / `shift` / `ctrl` / `alt` (= `opt` / `option`)。

`translate_prompt` 內可使用 `{target_lang}` 佔位符，執行時依來源語言自動代換為 `English` 或 `Traditional Chinese (zh-TW)`。

### Translate 流程（Shift+Cmd+R）

| 剪貼簿狀態 | 行為 |
|------|------|
| 有文字 | 直接翻譯（中→英 / 英→繁中），翻譯結果蓋回剪貼簿 |
| 只有圖片 | 先 OCR → OCR 文字寫回剪貼簿 → 通知 "OCR done, translating..." → 翻譯 → 翻譯結果蓋回剪貼簿 |
| 兩者都沒 | 通知 "No image or text in clipboard" |

語言判斷：算 Han 字元數 vs ASCII 字母數，Han × 4 > Latin 視為中文。

## 專案結構

```
go-ocr/
├── ReadMe.md
├── setup-cert.sh         # 一次性: 建立 stable codesign cert
├── build.sh              # 編譯 + 組 .app bundle + sign
├── run.sh                # dev: build + 直接執行
├── install.sh            # 正式: build + 裝到 /Applications [+ autostart]
├── assets/
│   └── icon.png          # 狀態列 icon (44x44 template image)
├── build/                # build.sh 產出 (gitignored)
└── src/
    ├── main.go           # 入口
    ├── tray.go           # 系統托盤 menu
    ├── config.go         # JSON 設定檔
    ├── hotkey.go         # 全域熱鍵
    ├── screenshot.go     # screencapture wrapper
    ├── clipboard.go      # 剪貼簿讀寫
    ├── ocr.go            # OpenAI-compatible HTTP
    ├── notify.go         # osascript 通知
    ├── flows.go          # 高階流程編排
    └── settings_window.go # Fyne 設定視窗
```
