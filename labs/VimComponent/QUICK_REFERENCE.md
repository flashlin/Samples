# 快速參考卡

## VimComponent Peer Dependencies 配置

### 📦 VimComponent (Library)

**package.json**
```json
{
  "peerDependencies": {
    "lit": "^3.0.0",
    "p5": "^1.6.0"
  }
}
```

**說明**：Library 使用 `peerDependencies` 告知使用者需要安裝的依賴

---

### 🚀 VimDemo (Application)

**package.json**
```json
{
  "dependencies": {
    "lit": "^3.3.1",
    "p5": "^1.11.10",
    "vimcomponent": "file:../VimComponent"
  }
}
```

**安裝指令**
```bash
pnpm add lit 'p5@^1.6.0'
```

**說明**：應用程式將 peer dependencies 安裝到 `dependencies`

---

## 常見指令

### 初次安裝
```bash
# 1. 建置 VimComponent
cd VimComponent
pnpm install
pnpm run build

# 2. 安裝 VimDemo（自動安裝 peer deps）
cd ../VimDemo
pnpm install
```

### 更新 VimComponent
```bash
# 1. 重新建置
cd VimComponent
pnpm run build

# 2. 更新 VimDemo
cd ../VimDemo
pnpm install
pnpm run dev
```

### 啟動開發
```bash
cd VimDemo
pnpm run dev
```

### 檢查版本
```bash
pnpm list lit p5 vimcomponent
```

### 修正 p5.js 版本問題
```bash
# 如果安裝了 p5 2.x，需要降級到 1.x
pnpm remove p5
pnpm add 'p5@^1.6.0'
```

---

## 問題排查

| 症狀 | 原因 | 解決方案 |
|------|------|---------|
| "Waiting for p5.js to load..." | 缺少 p5.js | `pnpm add 'p5@^1.6.0'` |
| "Waiting for p5.js to load..." | p5.js 版本是 2.x | `pnpm add 'p5@^1.6.0'` |
| peer dependency 警告 | 版本不匹配 | 按照警告安裝正確版本 |
| 編輯器空白 | 缺少 lit | `pnpm add lit` |

---

## 驗證清單

- [ ] VimComponent/package.json 有 `peerDependencies` 欄位
- [ ] VimDemo/package.json 的 dependencies 包含 lit 和 p5
- [ ] p5 版本是 1.x（如 1.11.10），不是 2.x
- [ ] `pnpm list` 顯示所有套件都已安裝
- [ ] 開發伺服器啟動後編輯器正常顯示

---

## 架構圖

```
VimComponent (Library)
├── package.json
│   ├── peerDependencies  ← 宣告需要 lit, p5
│   └── devDependencies   ← 開發時使用
└── dist/
    └── vim-editor.es.js

VimDemo (Application)  
├── package.json
│   └── dependencies      ← 安裝 lit, p5, vimcomponent
└── node_modules/
    ├── lit/              ← 自動安裝
    ├── p5/               ← 自動安裝
    └── vimcomponent/     ← 符號連結到 ../VimComponent
```

---

## 最佳實踐

1. ✅ Library 使用 `peerDependencies` 宣告共享依賴
2. ✅ Application 將所有依賴安裝到 `dependencies`
3. ✅ 使用 pnpm 7+ 可自動安裝 peer dependencies
4. ✅ 明確指定版本範圍（如 `^1.6.0`）
5. ⚠️ 注意 p5.js 必須使用 1.x 版本

