# 快速安裝指南

## 前置需求

- Node.js 18+ 
- pnpm 包管理工具

## 本地 Package 依賴說明

此專案使用 **本地 VimComponent package**，透過 pnpm 的 `file:` 協議安裝。

在 `package.json` 中的定義：
```json
{
  "dependencies": {
    "vimcomponent": "file:../VimComponent"
  }
}
```

這表示 VimComponent 必須位於此專案的父目錄的 `VimComponent` 資料夾中。

## 專案結構要求

```
labs/VimComponent/
├── VimComponent/          # VimComponent 原始專案
│   ├── dist/             # 建置後的檔案
│   │   ├── vim-editor.es.js
│   │   └── vim-editor.umd.js
│   └── package.json
└── VimDemo/              # 此示範專案
    ├── src/
    └── package.json
```

## 安裝步驟

### 1. 確保 VimComponent 已建置

```bash
cd ../VimComponent
pnpm install
pnpm run build
```

### 2. 安裝 VimDemo 依賴

```bash
cd ../VimDemo
pnpm install
```

pnpm 會：
1. 從 `../VimComponent` 建立符號連結到 `node_modules/vimcomponent`
2. 自動安裝 VimComponent 的 peer dependencies（lit 和 p5）

> **重要**：如果 pnpm 提示 peer dependency 警告，請確保安裝的是 p5@^1.6.0（1.x 版本），而非 2.x 版本：
> ```bash
> pnpm add lit 'p5@^1.6.0'
> ```

### 3. 啟動開發伺服器

```bash
pnpm run dev
```

## 使用本地 Package 的優點

✅ **即時更新**：修改 VimComponent 後重新建置，VimDemo 會自動使用最新版本  
✅ **無需發布**：不需要將 VimComponent 發布到 npm registry  
✅ **便於開發**：適合同時開發 library 和應用程式  
✅ **節省空間**：不會複製整個資料夾，只使用符號連結  

## 更新 VimComponent

當 VimComponent 有更新時：

```bash
# 1. 在 VimComponent 目錄重新建置
cd ../VimComponent
pnpm run build

# 2. 在 VimDemo 目錄重新安裝
cd ../VimDemo
pnpm install

# 3. 重啟開發伺服器
pnpm run dev
```

> **詳細資訊**：查看 [UPDATE_GUIDE.md](../UPDATE_GUIDE.md) 了解完整的更新流程、常見問題和最佳實踐。

## Peer Dependencies 詳解

### 什麼是 Peer Dependencies？

VimComponent 在 `package.json` 中宣告了 `peerDependencies`：

```json
{
  "peerDependencies": {
    "lit": "^3.0.0",
    "p5": "^1.6.0"
  }
}
```

這表示：
- 📦 VimComponent **需要**這些套件才能運作
- 🚫 VimComponent **不會**自動安裝這些套件
- ✅ 使用 VimComponent 的專案**必須**手動安裝這些套件

### 為什麼使用 Peer Dependencies？

1. **避免重複安裝**：如果您的專案已經使用 lit，就不需要再安裝一份
2. **版本一致性**：確保整個專案使用相同版本的依賴
3. **減少 bundle 大小**：不會打包重複的程式庫
4. **Library 最佳實踐**：這是開發可重用元件的標準做法

### p5.js 版本注意事項

⚠️ **必須使用 p5.js 1.x 版本**

- ✅ 支援：p5@^1.6.0（如 1.6.0、1.11.10 等）
- ❌ 不支援：p5@2.x（目前 VimComponent 尚未更新支援）

如果安裝了錯誤版本，會看到「Waiting for p5.js to load...」的訊息。

## 常見問題

### Q: 看到「Waiting for p5.js to load...」訊息
A: 表示缺少 p5.js 或版本不正確。執行：
```bash
pnpm add 'p5@^1.6.0'
```

### Q: pnpm 顯示 peer dependency 警告
A: 這是正常的。pnpm 會提示您需要安裝的 peer dependencies。按照提示安裝即可。

### Q: 安裝時出現找不到 VimComponent 的錯誤
A: 確認 VimComponent 資料夾位於正確的相對路徑 `../VimComponent`

### Q: 修改 VimComponent 後看不到變更
A: 需要在 VimComponent 目錄執行 `pnpm run build` 重新建置

### Q: 如何發布到生產環境？
A: 建議將 VimComponent 發布到 npm registry，然後修改 package.json 使用版本號而非本地路徑。使用者仍需手動安裝 peer dependencies。

