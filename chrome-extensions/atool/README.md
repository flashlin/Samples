# Chrome 擴充功能 - Vue 3 + TypeScript + Vite

這是一個使用 Vue 3、TypeScript 和 Vite 構建的 Chrome 擴充功能專案。

## 功能

- 點擊擴充功能圖標顯示彈出窗口
- 點擊按鈕顯示當前時間
- 使用 Chrome 存儲 API 保存上次點擊時間
- 在網頁中顯示浮動按鈕
- 獲取當前頁面信息

## 技術棧

- Vue 3 - 前端框架
- TypeScript - 類型安全的 JavaScript 超集
- Vite - 現代前端構建工具
- Chrome Extension API - 瀏覽器擴充功能 API

## 開發

### 安裝依賴

```bash
npm install
```

### 開發模式

```bash
npm run dev
```

### 構建擴充功能

```bash
npm run build:extension
```

構建完成後，擴充功能文件將位於 `dist` 目錄中。

## 安裝擴充功能

1. 在 Chrome 瀏覽器中打開 `chrome://extensions/`
2. 開啟右上角的「開發者模式」
3. 點擊「載入未封裝項目」
4. 選擇此專案的 `dist` 資料夾

## 文件結構

```
├── public/                # 靜態資源
│   ├── manifest.json      # 擴充功能配置文件
│   ├── popup.html         # 彈出窗口 HTML
│   ├── welcome.html       # 歡迎頁面
│   └── images/            # 圖標文件夾
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
├── src/                   # 源代碼
│   ├── popup/             # 彈出窗口
│   │   ├── App.vue        # 主組件
│   │   ├── index.html     # HTML 入口
│   │   ├── main.ts        # 入口文件
│   │   └── style.css      # 樣式文件
│   ├── background/        # 背景腳本
│   │   └── index.ts       # 背景腳本入口
│   ├── content/           # 內容腳本
│   │   └── index.ts       # 內容腳本入口
│   └── types/             # 類型定義
│       └── chrome.d.ts    # Chrome API 類型
├── scripts/               # 構建腳本
│   └── post-build.mjs     # 後構建處理
└── vite.config.ts         # Vite 配置
```

## 注意事項

- 此擴充功能需要 Chrome 88 或更高版本
- 使用了 Manifest V3 格式
