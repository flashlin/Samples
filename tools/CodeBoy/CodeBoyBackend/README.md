# CodeBoy Server

CodeBoy Server 是一個基於 ASP.NET Core 的 Web API 服務，提供從 Swagger 規格自動生成 C# 客戶端程式碼的功能。

## 功能特色

- 🔄 從 Swagger/OpenAPI 規格生成 C# 客戶端程式碼
- 🌐 RESTful API 介面
- 📚 自動生成的 API 文件 (Swagger UI)
- 🐳 Docker 容器化支持
- 🔍 健康檢查端點
- 📊 結構化日誌

## 快速開始

### 使用 Docker (推薦)

1. **建置 Docker 映像檔**
   ```bash
   ./build-docker.sh
   ```

2. **運行服務**
   ```bash
   ./run-docker.sh
   ```

3. **使用 Docker Compose**
   ```bash
   docker-compose up -d
   ```

### 本地開發

1. **還原套件**
   ```bash
   dotnet restore
   ```

2. **建置專案**
   ```bash
   dotnet build
   ```

3. **運行服務**
   ```bash
   dotnet run
   ```

## API 端點

### 程式碼生成

**POST** `/codegen/genWebApiClient`

生成 Web API 客戶端程式碼

**請求體**
```json
{
  "swaggerUrl": "https://example.com/swagger.json",
  "sdkName": "ExampleApi"
}
```

**回應**
```
生成的 C# 客戶端程式碼 (string)
```

### 健康檢查

**GET** `/health`

檢查服務狀態

**回應**
```json
{
  "status": "Healthy",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

## 服務端點

- **API 文件**: http://localhost:8080
- **健康檢查**: http://localhost:8080/health
- **程式碼生成**: http://localhost:8080/codegen/genWebApiClient

## Docker 指令

```bash
# 建置映像檔
docker build -t codeboy-server .

# 運行容器
docker run -p 8080:8080 codeboy-server

# 查看日誌
docker logs -f codeboy-server

# 停止容器
docker stop codeboy-server

# 移除容器
docker rm codeboy-server
```

## 環境變數

- `ASPNETCORE_ENVIRONMENT`: 執行環境 (Development/Production)
- `ASPNETCORE_URLS`: 服務監聽的 URL

## 架構

```
CodeBoyServer/
├── CodeBoyLib/           # 共用程式庫
│   ├── Models/          # 資料模型
│   └── Services/        # 業務邏輯服務
├── CodeBoyServer/       # Web API 專案
│   ├── ApiHandlers/     # API 處理器
│   ├── Models/          # API 模型
│   ├── Services/        # 應用服務
│   └── Program.cs       # 主程式
├── Dockerfile           # Docker 建置檔案
├── docker-compose.yml   # Docker Compose 配置
└── README.md           # 說明文件
```

## 開發說明

本專案使用：
- .NET 9.0
- ASP.NET Core Minimal APIs
- Swagger/OpenAPI 文件
- 依賴注入
- 結構化日誌

## 部署

### Docker 部署

1. 建置映像檔
2. 推送到容器註冊表
3. 在目標環境運行容器

### 雲端部署

支援部署到：
- Azure Container Instances
- AWS ECS
- Google Cloud Run
- Kubernetes

## 故障排除

**容器無法啟動**
- 檢查端口 8080 是否被占用
- 查看容器日誌: `docker logs codeboy-server`

**API 呼叫失敗**
- 檢查健康檢查端點是否正常
- 確認 Swagger URL 可以正常訪問

**建置失敗**
- 確保 .NET 9.0 SDK 已安裝
- 檢查專案參考是否正確
