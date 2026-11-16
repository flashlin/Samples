# Go-Youtu (Flutter YouTube 預覽、背景下載佇列與離線播放器)

遵循 **Clean Code** 與 **Clean Architecture** 原則設計的跨平台（Android & iOS）Dart / Flutter 行動應用程式。

## 核心功能與規格對應

1. **啟動直接開啟 YouTube 影片卡片最新預覽清單**
   - 免 API Key 自動抓取最新預覽影片，並支援關鍵字搜尋與下拉更新。
2. **完整的影片卡片下方操作列**
   - **訂閱 / 已訂閱**：點擊後直接訂閱該影片頻道，立即將該頻道的所有影片標示為「已訂閱優先」並加入 `[訂閱清單]`。
   - **標記已看過**：點擊後立即從 `[影片清單]` 移除，並紀錄至 `[已看過清單]`。
   - **Download**：點擊後卡片立即呈現**灰階半透明並顯示下載中圖示**，自動排入背景下載佇列 (`BackgroundDownloadScheduler`)。
   - **Play 播放 / Pause 暫停 / 倒退 30 秒 / 前進 30 秒 / 全螢幕播放**：點擊 Play 直接於卡片原位展開內嵌播放器（若已下載則自動使用本地離線 MP4 播放，否則串流線上來源），並支援切換橫向全螢幕播放。
   - **標記 Icon (離線 / 線上)**：清楚顯示每張卡片目前為「離線」或「線上」狀態。
   - **移除已下載檔 (保留紀錄)**：刪除手機端已下載的 MP4 實體檔案以釋放空間，同時保留 `hasEverDownloaded = true` 紀錄於 `[已下載過清單]`。
3. **主清單智慧排列**
   - **已下載優先**排在最上方 (`isOfflineAvailable == true`)。
   - **已訂閱的優先**排在次優先 (`isSubscribed == true`)。
   - **已看過的影片**自動移除不顯示於主清單。
4. **最多紀錄 10,000 筆資訊**
   - 本地 SQLite 資料庫 (`SqliteLocalDataSource`) 強制執行 `AppConstants.maxStoredRecords = 10000` 上限，超過時依更新時間自動淘汰最舊紀錄（FIFO）。
5. **最上方功能列**
   - `[影片清單]`、`[已下載過清單]`、`[訂閱清單]`、`[下載佇列清單]`、`[已看過清單]` 五大分頁即時切換與數量徽章。

---

## 快速啟動與編譯腳本

### 0. 自動安裝開發環境與 SDK (`setup.sh`)
若本機尚未安裝 Flutter / Dart SDK、OpenJDK 17、Android SDK / Command-line Tools 或 CocoaPods，可直接執行：
```bash
./setup.sh            # 自動檢查並安裝所有缺少的 SDK 與開發工具，並寫入 .env.sdk 與 ~/.zshrc
./setup.sh --check    # 僅檢查本機目前環境狀態，不執行安裝
```
> 註：若直接執行 `./dev.sh`、`./build-android.sh` 或 `./build-ios.sh` 時偵測到本機尚未安裝 Flutter SDK，也會自動呼叫 `./setup.sh` 完成環境安裝。

### 1. 開發啟動 (`dev.sh`)
預設以 **Android Debug** 模式啟動專案（若首次執行會自動檢查並補齊 Android/iOS 原生平台檔案）：
```bash
./dev.sh              # 啟動 Android Debug 模式 (flutter run --debug)
./dev.sh ios          # 啟動 iOS Debug 模式
./dev.sh test         # 執行 Clean Code 靜態檢查 (flutter analyze) 與單元測試 (flutter test)
./dev.sh clean        # 清理建置快取
```

### 2. 編譯 Android 執行程式 (`build-android.sh`)
```bash
./build-android.sh            # 預設編譯 Android Debug APK -> dist/android/go-youtu-debug.apk
./build-android.sh release    # 編譯 Android Release APK -> dist/android/go-youtu-release.apk
./build-android.sh aab        # 編譯 Android AppBundle -> dist/android/go-youtu-release.aab
```

### 3. 編譯 iOS 執行程式 (`build-ios.sh`)
```bash
./build-ios.sh                # 預設編譯 iOS Simulator Debug .app -> dist/ios/Runner-Simulator.app
./build-ios.sh release        # 編譯 iOS Release .app (免簽章) -> dist/ios/Runner-Release.app
./build-ios.sh ipa            # 編譯 iOS Archive IPA -> build/ios/ipa/
```
