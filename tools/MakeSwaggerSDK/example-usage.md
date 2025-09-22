# 使用範例

## 測試 Swagger 站點

您可以使用以下公開的 Swagger 站點來測試工具：

### 1. Petstore API (Swagger 官方範例)
```bash
dotnet run -- https://petstore.swagger.io/v2/swagger.json -n PetStore -o PetStoreClient.cs
```

### 2. OpenAPI Generator API
```bash
dotnet run -- https://generator.swagger.io/ -n OpenApiGenerator -o OpenApiGeneratorClient.cs
```

### 3. 本地 Swagger UI 範例
如果您有本地的 Swagger UI，例如：
```bash
dotnet run -- http://localhost:5000/swagger -n MyApi -o MyApiClient.cs
```

## 產生的程式碼使用方式

產生 SDK 後，您可以在專案中這樣使用：

### 1. 安裝必要的 NuGet 套件
```bash
dotnet add package Microsoft.Extensions.Http
dotnet add package Newtonsoft.Json
```

### 2. 在 Startup.cs 或 Program.cs 中註冊服務
```csharp
// .NET 6+ (Program.cs)
builder.Services.AddHttpClient();

// 或直接註冊您的 API 客戶端
builder.Services.AddHttpClient<PetStoreClient>(client =>
{
    client.BaseAddress = new Uri("https://petstore.swagger.io/v2/");
});
```

### 3. 在控制器或服務中使用
```csharp
public class PetService
{
    private readonly PetStoreClient _petStoreClient;
    
    public PetService(IHttpClientFactory httpClientFactory)
    {
        _petStoreClient = new PetStoreClient(httpClientFactory, "https://petstore.swagger.io/v2");
    }
    
    public async Task<List<Pet>> GetAvailablePets()
    {
        return await _petStoreClient.FindPetsByStatus("available");
    }
}
```

## 注意事項

1. **SSL 憑證**: 確保 Swagger URL 的 SSL 憑證有效
2. **CORS 設定**: 本地測試時可能需要處理 CORS 問題
3. **認證**: 如果 API 需要認證，您需要手動添加認證邏輯
4. **複雜型別**: 對於複雜的資料型別，可能需要手動調整產生的 DTO 類別

## 故障排除

### 無法連接到 Swagger URL
- 檢查網址是否正確
- 確認網路連接
- 檢查防火牆設定

### 無法找到 Swagger JSON
- 確認網站確實是 Swagger UI
- 檢查是否有多個可能的 JSON 端點
- 手動查看頁面原始碼確認 JSON URL

### 產生的程式碼有編譯錯誤
- 檢查命名空間是否正確
- 確認所有必要的 using 語句已包含
- 檢查自訂型別是否需要額外定義
