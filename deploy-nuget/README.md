# NuGet 部署腳本

這個腳本用於自動化部署 .NET 專案到 NuGet。

## 功能

- 使用 fzf 互動式選擇要部署的專案
- 自動讀取 .env 檔案中的環境變數
- 自動建置、打包和發布到 NuGet

## 前置需求

1. **fzf** - 互動式選擇器
   ```bash
   # macOS
   brew install fzf
   
   # Ubuntu/Debian
   sudo apt install fzf
   ```

2. **.NET SDK** - 用於建置和發布
   ```bash
   # 下載並安裝 .NET SDK
   # https://dotnet.microsoft.com/download
   ```

## 設定

### 1. 建立 .env 檔案

在腳本目錄中建立 `.env` 檔案：

```bash
# NuGet API Key (required for publishing)
NUGET_API_KEY=your_nuget_api_key_here

# NuGet Source URL
NUGET_SOURCE=https://api.nuget.org/v3/index.json

# Build Configuration
BUILD_CONFIGURATION=Release

# Output Directory for nupkg files
NUPKG_OUTPUT_DIR=./nupkg
```

### 2. 取得 NuGet API Key

1. 登入 [NuGet.org](https://www.nuget.org)
2. 前往 Account Settings
3. 在 API Keys 區段建立新的 API Key
4. 將 API Key 複製到 `.env` 檔案中

## 使用方法

1. 確保腳本有執行權限：
   ```bash
   chmod +x deploy.sh
   ```

2. 執行腳本：
   ```bash
   ./deploy.sh
   ```

3. 使用 fzf 選擇要部署的專案

## 專案結構

腳本假設專案結構如下：

```
deploy-nuget/
├── deploy.sh
├── .env
└── README.md

../T1.Standard/
├── T1.Standard.csproj
└── ...

../T1.Slack.SDK/
├── T1.Slack.SDK.csproj
└── ...
```

## 環境變數說明

| 變數名稱 | 預設值 | 說明 |
|---------|--------|------|
| `NUGET_API_KEY` | 無 | NuGet API Key，用於發布 |
| `NUGET_SOURCE` | `https://api.nuget.org/v3/index.json` | NuGet 來源 URL |
| `BUILD_CONFIGURATION` | `Release` | 建置配置 |
| `NUPKG_OUTPUT_DIR` | `./nupkg` | nupkg 檔案輸出目錄 |

## 注意事項

- 腳本會自動檢查 `.env` 檔案是否存在
- 如果沒有設定 `NUGET_API_KEY`，會嘗試不使用 API Key 發布
- 確保目標專案目錄存在且包含對應的 `.csproj` 檔案 