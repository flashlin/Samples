# CodeBoy Frontend

一個使用 Vue3 + TypeScript + Tailwind CSS 建置的現代化前端專案。

## 🚀 技術棧

- **框架**: Vue 3 (Composition API)
- **語言**: TypeScript
- **樣式**: Tailwind CSS 3 (支援暗黑模式)
- **狀態管理**: Pinia
- **HTTP 客戶端**: Axios
- **建置工具**: Vite
- **Mock 工具**: Mock.js + vite-plugin-mock
- **測試框架**: Jest + Vue Test Utils
- **程式品質**: ESLint + Prettier

## 📦 依賴安裝

確保使用 Node.js 22.16.0：

```bash
nvm use 22.16.0
pnpm install
```

## 🛠️ 開發指令

```bash
# 啟動開發伺服器
pnpm dev

# 建置生產版本
pnpm build

# 預覽生產建置
pnpm preview

# 執行 Linting
pnpm lint

# 格式化程式碼
pnpm format

# 執行測試
pnpm test

# 監聽模式執行測試
pnpm test:watch
```

## 📁 專案結構

```
CodeBoyFront/
├─ public/                  # 靜態資源
├─ mock/                    # Mock API 資料
│  ├─ app.js                # 應用相關 API
│  └─ user.js               # 使用者相關 API  
├─ src/
│  ├─ apis/                 # API 封裝
│  │  ├─ request.ts         # Axios 設定
│  │  └─ user.ts            # 使用者 API
│  ├─ assets/               # 資源檔案
│  │  └─ main.css           # 全域樣式
│  ├─ components/           # 共用元件
│  ├─ stores/               # Pinia 狀態管理
│  │  └─ user.ts            # 使用者狀態
│  ├─ views/                # 頁面元件
│  ├─ App.vue               # 根元件
│  └─ main.ts               # 應用入口
├─ .env.development         # 開發環境變數
├─ .env.production          # 生產環境變數
├─ .eslintrc.cjs            # ESLint 設定
├─ .prettierrc              # Prettier 設定
├─ jest.config.js           # Jest 測試設定
├─ tailwind.config.js       # Tailwind 設定
├─ tsconfig.app.json        # TypeScript 設定
└─ vite.config.ts           # Vite 設定
```

## 🎨 特色功能

### ✅ 暗黑模式支援
- 使用 Tailwind CSS 的 `dark:` 前綴
- 自動套用暗黑主題

### ✅ API Mock 系統
- 開發時自動載入 Mock API
- 支援動態數據生成

### ✅ 類型安全
- 完整的 TypeScript 支援
- API 請求/回應類型定義

### ✅ 現代化開發體驗
- 熱模組重載 (HMR)
- 自動程式碼格式化
- 即時 Linting

### ✅ 測試就緒
- Jest 單元測試框架
- Vue Test Utils 元件測試

## 🌐 開發伺服器

啟動後可在 http://localhost:5173 查看應用程式

## 📝 環境變數

### 開發環境 (.env.development)
```
VITE_APP_API_BASE_URL=http://localhost:3000/api
VITE_APP_TITLE=CodeBoy Frontend (Dev)
```

### 生產環境 (.env.production)
```
VITE_APP_API_BASE_URL=/api
VITE_APP_TITLE=CodeBoy Frontend
```

## 🧪 Mock API

### 使用者 API
- `POST /api/user/login` - 使用者登入
- `GET /api/user/profile` - 取得使用者資料

### 應用 API
- `GET /api/app/health` - 健康檢查
- `GET /api/app/info` - 應用資訊

## 🎯 開發建議

1. 使用 Composition API 編寫元件
2. 遵循 TypeScript 嚴格模式
3. 使用 Pinia 管理複雜狀態
4. 利用 Tailwind 的實用類別優先方法
5. 為重要功能編寫單元測試

## 📄 授權

此專案為私人專案，僅供學習和開發使用。