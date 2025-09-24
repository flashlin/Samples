# MakeSwaggerSDK

一個從 Swagger UI 網頁自動產生 C# SDK 的命令列工具。

## 功能特點

- 解析 Swagger UI 網頁並提取 Swagger JSON 規格
- 分析所有 API 端點及其參數
- 自動產生完整的 C# HTTP 客戶端程式碼
- 支援各種 HTTP 方法 (GET, POST, PUT, DELETE, PATCH 等)
- 處理路徑參數、查詢參數、請求體和標頭參數
- 生成型別安全的方法簽名
- 支援依賴注入 (IHttpClientFactory)

## 使用方式

### Docker 方式 (推薦)

#### 互動式管理
```bash
# 使用 fzf 選擇 start 或 stop
./run-docker.sh
```

#### 直接建置和運行
```bash
# 建置 Docker 映像檔
./build-docker.sh

# 使用 Docker Compose
docker-compose up -d
```

### 本地開發

#### 基本用法

```bash
cd CodeGen
dotnet run -- http://your-swagger-url
```

#### 進階用法

```bash
# 指定 SDK 名稱
dotnet run -- http://your-swagger-url -n MyApi

# 指定輸出檔案名稱
dotnet run -- http://your-swagger-url -o MyApiClient.cs

# 完整範例
dotnet run -- http://your-swagger-url -n MyApi -o MyApiClient.cs
```

### 命令列參數

- `swagger-url` (必需): Swagger UI 的網址
- `-n, --name`: SDK 類別名稱 (預設: "ApiClient")
- `-o, --output`: 輸出檔案名稱 (預設: "{SdkName}.cs")

## 輸出檔案結構

產生的 SDK 包含：

1. **DTO 類別**: 用於 API 回應的資料傳輸物件
2. **客戶端類別**: 主要的 HTTP 客戶端，包含所有 API 方法
3. **建構函式**: 支援 IHttpClientFactory 和直接 HttpClient 注入
4. **型別安全方法**: 每個 API 端點對應一個方法

## 範例產生的程式碼

```csharp
public class MyApiClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public MyApiClient(IHttpClientFactory httpClientFactory, string baseUrl)
    {
        _httpClient = httpClientFactory.CreateClient();
        _baseUrl = baseUrl.TrimEnd('/');
    }

    public async Task<List<User>> GetUsers(int? limit = null, int? offset = null)
    {
        var url = "/users";
        var queryParams = new List<string>();
        
        if (limit != null)
            queryParams.Add($"limit={Uri.EscapeDataString(limit.ToString())}");
        if (offset != null)
            queryParams.Add($"offset={Uri.EscapeDataString(offset.ToString())}");
            
        if (queryParams.Any())
            url += "?" + string.Join("&", queryParams);

        var request = new HttpRequestMessage(HttpMethod.GET, _baseUrl + url);
        var response = await _httpClient.SendAsync(request);
        response.EnsureSuccessStatusCode();

        var responseContent = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<List<User>>(responseContent) ?? new List<User>();
    }
}
```

## 專案結構

```
MakeSwaggerSDK/
├── Models/
│   └── SwaggerEndpoint.cs    # 資料模型定義
├── Services/
│   ├── SwaggerUiParser.cs    # Swagger UI 解析器
│   └── SwaggerClientCodeGenerator.cs  # 程式碼產生器
├── Program.cs                # 主程式進入點
└── MakeSwaggerSDK.csproj    # 專案檔
```

## 依賴套件

- **CommandLineParser**: 命令列參數解析
- **HtmlAgilityPack**: HTML 內容解析
- **Newtonsoft.Json**: JSON 序列化/反序列化

## 注意事項

1. 確保 Swagger URL 可以正常存取
2. 產生的程式碼可能需要手動調整複雜的資料型別
3. 建議在使用前先檢查產生的程式碼
4. 對於大型 API，產生的檔案可能會很大

## 建置專案

```bash
cd MakeSwaggerSDK
dotnet build
```

## 執行範例

```bash
# 假設有一個 Swagger UI 在 http://localhost:5000/swagger
dotnet run -- http://localhost:5000/swagger -n PetStore -o PetStoreClient.cs
```

這會產生一個名為 `PetStoreClient.cs` 的檔案，包含 `PetStoreClient` 類別及所有必要的 API 方法。
