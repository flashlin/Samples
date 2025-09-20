# DemoSDK - gRPC Client SDK

這個 SDK 提供了自動產生的 gRPC Client 設定和擴充方法。

## 使用方式

### 1. 安裝套件參考

```xml
<ProjectReference Include="path/to/DemoSDK/DemoSDK.csproj" />
```

### 2. 設定相依性注入

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using DemoServer.Services; // 來自 MyProtoContracts 的自動產生程式碼

var services = new ServiceCollection();

// 設定 gRPC 伺服器位址
services.Configure<GreeterGrpcConfig>(config =>
{
    config.ServerUrl = "https://your-grpc-server:7001";
});

// 使用自動產生的擴充方法註冊 gRPC Client
services.AddGreeterGrpcSdk();

var serviceProvider = services.BuildServiceProvider();
```

### 3. 使用 gRPC Client

```csharp
// 注入並使用包裝的 gRPC Client（推薦）
var grpcClient = serviceProvider.GetService<IGreeterGrpcClient>();
var response = await grpcClient.SayHelloAsync(new HelloRequestGrpcDto 
{ 
    Name = "World" 
});

Console.WriteLine($"Response: {response.Message}");
```

## 自動產生的類別

此 SDK 透過 Source Generator 自動產生以下類別：

- `GreeterGrpcConfig` - gRPC 伺服器設定
- `GreeterGrpcExtension` - 相依性注入擴充方法
- `IGreeterGrpcClient` - gRPC Client 介面
- `GreeterGrpcClient` - gRPC Client 實作

所有這些類別都位於 `DemoServer.Services` 命名空間中。
