# Peer Dependencies 配置指南

本指南說明如何正確配置 VimComponent 的 peer dependencies，避免「Waiting for p5.js to load...」的問題。

## 問題背景

當 VimComponent 被其他專案使用時，如果缺少必要的依賴（lit 和 p5.js），編輯器將無法正常載入。

## 解決方案：使用 Peer Dependencies

### 1. VimComponent 配置

在 `VimComponent/package.json` 中新增 `peerDependencies` 欄位：

```json
{
  "name": "vimcomponent",
  "version": "1.0.0",
  "peerDependencies": {
    "lit": "^3.0.0",
    "p5": "^1.6.0"
  },
  "devDependencies": {
    "@types/p5": "^1.7.3",
    "lit": "^3.1.1",
    "p5": "^1.6.0",
    ...
  }
}
```

**重點說明**：
- `peerDependencies`：告知使用者必須安裝的依賴
- `devDependencies`：保留用於開發和測試 VimComponent 本身

### 2. 使用專案（如 VimDemo）配置

在使用 VimComponent 的專案中，必須安裝 peer dependencies：

```bash
pnpm add lit 'p5@^1.6.0'
```

`package.json` 會自動更新：

```json
{
  "dependencies": {
    "lit": "^3.3.1",
    "p5": "^1.11.10",
    "vimcomponent": "file:../VimComponent",
    "vue": "^3.5.22"
  }
}
```

## Peer Dependencies 的優點

### ✅ 1. 避免重複安裝
如果專案已經使用 lit 或 p5.js，不會再安裝一份副本。

### ✅ 2. 版本一致性
確保整個專案使用相同版本的共享依賴。

### ✅ 3. 減少 Bundle 大小
避免在最終 bundle 中包含重複的程式庫。

### ✅ 4. 明確的依賴關係
使用者清楚知道需要安裝哪些依賴。

### ✅ 5. 靈活性
使用者可以在相容範圍內選擇特定版本。

## pnpm 的 Peer Dependencies 處理

### 自動安裝（pnpm 7+）

pnpm v7 及以上版本會**自動安裝** peer dependencies：

```bash
cd VimDemo
pnpm install  # 自動安裝 lit 和 p5
```

### 警告訊息

如果版本不匹配，pnpm 會顯示警告：

```
WARN Issues with peer dependencies found
.
└─┬ vimcomponent 1.0.0
  └── ✕ unmet peer p5@^1.6.0: found 2.0.5
```

### 手動安裝

也可以手動安裝特定版本：

```bash
pnpm add lit 'p5@^1.6.0'
```

## p5.js 版本要求

### ⚠️ 重要：必須使用 1.x 版本

VimComponent 目前支援 p5.js 1.x 版本，不支援 2.x 版本。

#### ✅ 正確版本
```bash
pnpm add 'p5@^1.6.0'
# 會安裝 1.11.10 或其他 1.x 最新版本
```

#### ❌ 錯誤版本
```bash
pnpm add p5
# 可能會安裝 2.0.5，導致編輯器無法載入
```

### 檢查已安裝版本

```bash
pnpm list p5
```

輸出應該顯示 1.x 版本：
```
p5 1.11.10
```

## 驗證安裝

### 1. 檢查套件列表

```bash
cd VimDemo
pnpm list lit p5 vimcomponent
```

預期輸出：
```
dependencies:
lit 3.3.1
p5 1.11.10
vimcomponent file:../VimComponent(lit@3.3.1)(p5@1.11.10)
```

### 2. 檢查 node_modules

```bash
ls -la node_modules/ | grep -E 'lit|p5|vimcomponent'
```

應該看到三個目錄都存在。

### 3. 執行開發伺服器

```bash
pnpm run dev
```

編輯器應該正常載入，不會顯示「Waiting for p5.js to load...」。

## 發布到 npm Registry

如果將 VimComponent 發布到 npm，使用者安裝時：

```bash
npm install vimcomponent
```

npm/pnpm 會提示安裝 peer dependencies：

```
npm WARN vimcomponent@1.0.0 requires a peer of lit@^3.0.0 but none is installed.
npm WARN vimcomponent@1.0.0 requires a peer of p5@^1.6.0 but none is installed.
```

使用者需要執行：

```bash
npm install lit p5@^1.6.0
```

## 總結

| 方面 | dependencies | peerDependencies | devDependencies |
|------|-------------|------------------|-----------------|
| 誰使用 | 執行時需要 | 執行時需要，但由使用者提供 | 僅開發時需要 |
| 自動安裝 | ✅ 是 | ⚠️ 視 package manager 而定 | ❌ 否 |
| 適用場景 | 應用程式依賴 | Library 的共享依賴 | 開發工具 |
| VimComponent | - | lit, p5 | vite, typescript 等 |

**最佳實踐**：
- Library（如 VimComponent）應該使用 `peerDependencies` 宣告執行時依賴
- 應用程式（如 VimDemo）應該將所有依賴放在 `dependencies`
- 開發工具永遠放在 `devDependencies`

