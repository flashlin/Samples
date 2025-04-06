# T1.GrpcSourceGenerator

這是一個 C# Source Generator，可以自動從標記了 `[GenerateGrpcService]` 特性的類別生成 gRPC 服務實現和 .proto 文件內容。

## 功能

1. 分析標記了 `[GenerateGrpcService(typeof(IYourInterface))]` 的類別
2. 基於接口定義自動生成 .proto 文件內容
3. 自動生成 gRPC 服務實現類，將請求轉發到實際的接口實現

## 使用方法

### 步驟 1：添加 Source Generator 引用

在您的專案中添加對 Source Generator 的引用：

```xml
<ItemGroup>
  <ProjectReference Include="..\T1.GrpcSourceGenerator\T1.GrpcSourceGenerator.csproj" 
                    OutputItemType="Analyzer" 
                    ReferenceOutputAssembly="true" />
</ItemGroup>
```

### 步驟 2：定義您的接口

```csharp
public interface IGreeter
{
    Task<string> SayHelloAsync(string name);
    Task<UserInfo> GetUserInfoAsync(int id, bool includeDetails);
    int Add(int a, int b);
}

public class UserInfo
{
    public int Id { get; set; }
    public string Name { get; set; }
    public int Age { get; set; }
    public string Details { get; set; }
}
```

### 步驟 3：實現接口並添加特性

```csharp
[GenerateGrpcService(typeof(IGreeter))]
public class MyGreeter : IGreeter
{
    public Task<string> SayHelloAsync(string name)
    {
        return Task.FromResult($"Hello, {name}!");
    }
    
    public Task<UserInfo> GetUserInfoAsync(int id, bool includeDetails)
    {
        var userInfo = new UserInfo
        {
            Id = id,
            Name = $"User-{id}",
            Age = 30 + id % 20,
            Details = includeDetails ? "詳細信息..." : null
        };
        
        return Task.FromResult(userInfo);
    }
    
    public int Add(int a, int b)
    {
        return a + b;
    }
}
```

### 步驟 4：查看生成的代碼

編譯後，Source Generator 會生成：

1. `.proto` 文件內容（作為註釋添加到生成的源文件中）
2. 消息類型（`*RequestMessage`，`*ReplyMessage`）
3. gRPC 服務基類（`*Base`）
4. gRPC 服務實現類（`*GrpcService`）

### 步驟 5：在 ASP.NET Core 中使用生成的服務

```csharp
var builder = WebApplication.CreateBuilder(args);

// 添加 gRPC 服務
builder.Services.AddGrpc();

// 註冊您的接口實現
builder.Services.AddSingleton<IGreeter, MyGreeter>();

// 註冊生成的 gRPC 服務
builder.Services.AddSingleton<GreeterGrpcService>();

var app = builder.Build();

// 映射 gRPC 服務
app.MapGrpcService<GreeterGrpcService>();

app.Run();
```

## 注意事項

1. 生成的代碼僅在編譯時存在於內存中，您可以在編譯後的程序集中找到生成的類型。
2. 為了實際使用 gRPC，您需要將生成的 .proto 文件內容複製到實際的 .proto 文件中，並使用 Grpc.Tools 進行編譯。
3. 轉換邏輯已自動生成，但在複雜場景下可能需要進行調整。

## 限制

1. 目前不支持流式 gRPC 方法（Streaming）。
2. 複雜類型的轉換需要手動實現。
3. .proto 文件需要手動配置和編譯。

---

這個 Source Generator 可以大大減少開發 gRPC 服務時的重複代碼，使您可以專注於業務邏輯實現，而不是手動編寫 gRPC 服務樣板代碼。 