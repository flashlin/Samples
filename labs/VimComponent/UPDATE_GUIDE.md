# VimComponent 更新指南

當 VimComponent 有更新時，如何讓使用專案（如 VimDemo）使用最新版本。

## 情境說明

VimDemo 使用 `file:` 協議安裝本地的 VimComponent：

```json
{
  "dependencies": {
    "vimcomponent": "file:../VimComponent"
  }
}
```

pnpm 會建立符號連結到 `../VimComponent`，所以當 VimComponent 更新時，需要確保變更被正確應用。

---

## 快速更新步驟

### 方法 1：重新安裝（推薦）

```bash
# 1. 在 VimComponent 目錄重新建置
cd /path/to/VimComponent
pnpm run build

# 2. 在 VimDemo 目錄重新安裝
cd ../VimDemo
pnpm install

# 3. 重啟開發伺服器
pnpm run dev
```

### 方法 2：使用 pnpm update

```bash
# 1. 在 VimComponent 目錄重新建置
cd /path/to/VimComponent
pnpm run build

# 2. 在 VimDemo 目錄更新 VimComponent
cd ../VimDemo
pnpm update vimcomponent

# 3. 重啟開發伺服器
pnpm run dev
```

### 方法 3：清除並重裝（完全重置）

```bash
# 1. 在 VimComponent 目錄重新建置
cd /path/to/VimComponent
pnpm run build

# 2. 在 VimDemo 目錄移除並重裝
cd ../VimDemo
rm -rf node_modules
pnpm install

# 3. 重啟開發伺服器
pnpm run dev
```

---

## 詳細說明

### 步驟 1：建置 VimComponent

**為什麼需要？**
- VimComponent 的原始碼在 `src/` 目錄
- 需要編譯成 JavaScript 到 `dist/` 目錄
- VimDemo 使用的是 `dist/` 中的編譯後檔案

```bash
cd /path/to/VimComponent
pnpm run build
```

**輸出範例**：
```
> vimcomponent@1.0.0 build
> tsc && vite build

vite v4.5.14 building for production...
✓ 19 modules transformed.
dist/vim-editor.es.js  79.32 kB │ gzip: 18.70 kB
dist/vim-editor.umd.js  59.91 kB │ gzip: 16.86 kB
✓ built in 268ms
```

### 步驟 2：更新 VimDemo 的依賴

**為什麼需要？**

雖然使用符號連結，但 pnpm 會：
1. 檢查 package.json 的變更
2. 重新建立符號連結
3. 更新 lockfile

```bash
cd /path/to/VimDemo
pnpm install
```

**pnpm install 做了什麼？**
- 讀取 `package.json`
- 檢查 `file:../VimComponent` 
- 重新建立 `node_modules/vimcomponent` 符號連結
- 確保使用最新的 `dist/` 檔案

### 步驟 3：重啟開發伺服器

**為什麼需要？**
- Vite 可能緩存了舊的模組
- 需要重新載入更新後的 VimComponent

```bash
# 如果開發伺服器正在運行，先停止（Ctrl+C）
# 然後重啟
pnpm run dev
```

---

## 驗證更新

### 1. 檢查符號連結

```bash
cd VimDemo
ls -la node_modules/vimcomponent
```

應該看到類似：
```
lrwxr-xr-x  vimcomponent -> ../../VimComponent
```

### 2. 檢查 dist 檔案時間戳

```bash
ls -la node_modules/vimcomponent/dist/
```

時間戳應該是最新的建置時間。

### 3. 檢查瀏覽器 Console

打開開發者工具（F12），應該看到：
```
p5.js loaded, initializing...
p5 setup called
Canvas size: 800 600
```

**不應該看到**：
```
Waiting for p5.js to load...
```

### 4. 檢查 package 版本

```bash
cd VimDemo
pnpm list vimcomponent
```

輸出：
```
vimcomponent file:../VimComponent(lit@3.3.1)(p5@1.11.10)
```

---

## 常見問題

### Q: 為什麼 pnpm install 後還是用舊版本？

A: 可能是因為：
1. **忘記重新建置 VimComponent**
   ```bash
   cd VimComponent
   pnpm run build  # 必須先執行這個！
   ```

2. **Vite 緩存問題**
   ```bash
   cd VimDemo
   rm -rf node_modules/.vite  # 清除 Vite 緩存
   pnpm run dev
   ```

3. **瀏覽器緩存**
   - 開啟開發者工具
   - 右鍵點擊重新整理按鈕
   - 選擇「清除快取並強制重新整理」

### Q: 符號連結是否會自動更新？

A: 部分會：
- ✅ **符號連結本身**：指向 `../VimComponent`，自動連結到該目錄
- ❌ **編譯後的 dist 檔案**：需要重新 `pnpm run build`
- ⚠️ **開發伺服器**：需要重啟以重新載入模組

### Q: 什麼時候需要清除 node_modules？

A: 通常不需要，但以下情況建議清除：
- pnpm 版本更新後
- package.json 結構大幅變更
- 出現奇怪的依賴問題
- 想要完全重置

### Q: 可以自動化更新流程嗎？

A: 可以！建立一個 script：

```bash
# update-and-run.sh
#!/bin/bash

echo "Building VimComponent..."
cd VimComponent
pnpm run build

echo "Updating VimDemo..."
cd ../VimDemo
pnpm install

echo "Starting dev server..."
pnpm run dev
```

使用：
```bash
chmod +x update-and-run.sh
./update-and-run.sh
```

### Q: 如何知道 VimComponent 是否有更新？

A: 檢查方式：
1. **Git 狀態**：
   ```bash
   cd VimComponent
   git log -1  # 查看最新 commit
   ```

2. **檔案時間戳**：
   ```bash
   ls -la dist/
   ```

3. **版本號**：
   ```bash
   grep version package.json
   ```

---

## 開發工作流程

### 同時開發 VimComponent 和 VimDemo

```bash
# Terminal 1: 監視 VimComponent 變更並自動建置
cd VimComponent
pnpm run build --watch  # 如果支援 watch mode

# Terminal 2: 運行 VimDemo 開發伺服器
cd VimDemo
pnpm run dev
```

如果 VimComponent 不支援 watch mode，可以使用 `nodemon`：

```bash
# 安裝 nodemon
pnpm add -g nodemon

# 監視原始碼變更並自動建置
cd VimComponent
nodemon --watch src --exec "pnpm run build"
```

### 最佳實踐

1. **頻繁建置**：修改 VimComponent 後立即建置
2. **版本控制**：使用 Git tag 標記重要版本
3. **變更日誌**：在 CHANGELOG.md 記錄變更
4. **語義化版本**：更新 package.json 的版本號

---

## pnpm 本地 Package 特性

### file: 協議的優點

```json
{
  "dependencies": {
    "vimcomponent": "file:../VimComponent"
  }
}
```

**優點**：
- ✅ 即時反映變更（重新建置後）
- ✅ 不需要發布到 npm
- ✅ 適合本地開發和測試
- ✅ 節省空間（符號連結）
- ✅ 版本控制友善

### 符號連結機制

pnpm 使用硬連結和符號連結來節省空間：

```
VimDemo/
└── node_modules/
    └── vimcomponent -> ../../VimComponent  # 符號連結
```

這意味著：
- `VimComponent` 的變更會立即反映
- 但需要重新建置 dist 檔案
- 需要重啟開發伺服器

---

## 發布到生產環境

當準備好發布時，應該：

### 選項 1：發布到 npm

```bash
# 1. 更新版本號
cd VimComponent
npm version patch  # 或 minor, major

# 2. 發布到 npm
npm publish

# 3. 更新 VimDemo 使用 npm 版本
cd ../VimDemo
# 修改 package.json
{
  "dependencies": {
    "vimcomponent": "^1.0.0"  // 使用版本號而非 file:
  }
}
pnpm install
```

### 選項 2：使用 Git 依賴

```json
{
  "dependencies": {
    "vimcomponent": "git+https://github.com/username/vimcomponent.git"
  }
}
```

### 選項 3：私有 npm registry

適合公司內部使用：
```bash
npm publish --registry https://your-private-registry.com
```

---

## 總結

### 標準更新流程

```bash
# 1. 建置 VimComponent
cd VimComponent && pnpm run build

# 2. 更新 VimDemo
cd ../VimDemo && pnpm install

# 3. 重啟
pnpm run dev
```

### 記住這三個關鍵點

1. 📦 **Build First**: 先建置 VimComponent
2. 🔄 **Reinstall**: 重新安裝以更新連結
3. 🚀 **Restart**: 重啟伺服器載入新版本

就這麼簡單！✨

