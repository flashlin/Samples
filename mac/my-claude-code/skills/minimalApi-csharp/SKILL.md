# Minimal API 最佳實踐

## 程式碼風格規範

### 基本結構

使用靜態類別和擴充方法來組織 Minimal API endpoints：

```csharp
public static class xxxApiHandler
{
    public static IEndpointRouteBuilder MapXxxEndpoints(
        this IEndpointRouteBuilder endpoints,
        string prefix = "api/xxx")
    {
        var group = endpoints.MapGroup($"/{prefix}");

        group.MapPost("/action", HandleXxxRequest)
            .WithName("XxxAction")
            .Produces<XxxResponse>(200)
            .Produces(400)
            .Produces(404)
            .Produces(504);

        return endpoints;
    }

    private static async Task<XxxResponse> HandleXxxRequest(
        XxxRequest request,
        IXxxUseCase xxxUseCase)
    {
        return xxxUseCase.ExecuteAction(request);
    }
}
```

## 核心原則

### 1. 明確的輸入和返回型別

- **必須明確宣告** 所有 endpoint 的請求和回應型別
- 不使用匿名型別或動態型別
- 使用 `.Produces<T>()` 標註所有可能的回應狀態碼

### 2. 例外處理策略

- **預設不使用 try-catch**
- 讓例外向上層傳播，由全域例外處理器統一處理
- **只有在需要特殊處理時才使用 try-catch**

```csharp
// ❌ 不推薦：不必要的 try-catch
private static async Task<UserResponse> HandleGetUser(
    int id,
    IUserUseCase userUseCase)
{
    try
    {
        return await userUseCase.GetUser(id);
    }
    catch (Exception ex)
    {
        throw; // 只是重新拋出，沒有意義
    }
}

// ✅ 推薦：直接呼叫，讓例外傳播
private static async Task<UserResponse> HandleGetUser(
    int id,
    IUserUseCase userUseCase)
{
    return await userUseCase.GetUser(id);
}

// ✅ 可接受：有特殊處理邏輯時才使用
private static async Task<Result<UserResponse>> HandleGetUser(
    int id,
    IUserUseCase userUseCase,
    ILogger<UserApiHandler> logger)
{
    try
    {
        var user = await userUseCase.GetUser(id);
        return Result.Success(user);
    }
    catch (UserNotFoundException ex)
    {
        logger.LogWarning(ex, "使用者不存在: {UserId}", id);
        return Result.Failure<UserResponse>("使用者不存在");
    }
}
```

### 3. 職責分離

- Handler 方法只負責接收參數並呼叫 UseCase
- 業務邏輯放在 UseCase 層
- Handler 保持簡潔，避免複雜邏輯

### 4. 命名規範

- Handler 類別命名：`{功能}ApiHandler`
- 擴充方法命名：`Map{功能}Endpoints`
- Handler 方法命名：`Handle{動作}{資源}`

## 完整範例

```csharp
public static class ReverseProxyApiHandler
{
    public static IEndpointRouteBuilder MapReverseProxyEndpoints(
        this IEndpointRouteBuilder endpoints,
        string prefix = "api/reverse")
    {
        var group = endpoints.MapGroup($"/{prefix}");

        group.MapPost("/request", HandleReverseProxyRequest)
            .WithName("ReverseProxyRequest")
            .Produces<ReverseProxyResponse>(200)
            .Produces(400)
            .Produces(404)
            .Produces(504);

        group.MapGet("/status/{id}", HandleGetProxyStatus)
            .WithName("GetProxyStatus")
            .Produces<ProxyStatusResponse>(200)
            .Produces(404);

        return endpoints;
    }

    private static async Task<ReverseProxyResponse> HandleReverseProxyRequest(
        ReverseProxyRequest request,
        IReverseProxyUseCase reverseProxyUseCase)
    {
        return await reverseProxyUseCase.ExecuteProxy(request);
    }

    private static async Task<ProxyStatusResponse> HandleGetProxyStatus(
        string id,
        IReverseProxyUseCase reverseProxyUseCase)
    {
        return await reverseProxyUseCase.GetProxyStatus(id);
    }
}
```

## 註冊方式

在 `Program.cs` 中註冊：

```csharp
var builder = WebApplication.CreateBuilder(args);

// 註冊服務
builder.Services.AddScoped<IReverseProxyUseCase, ReverseProxyUseCase>();

var app = builder.Build();

// 註冊 endpoints
app.MapReverseProxyEndpoints();

app.Run();
```

## 注意事項

- 不在程式碼中寫註解，使用有意義的方法名稱替代
- 方法不要過長，依照功能分解為更小方法
- 保持 Handler 方法簡潔，複雜邏輯移至 UseCase
- 使用依賴注入提供服務實例
