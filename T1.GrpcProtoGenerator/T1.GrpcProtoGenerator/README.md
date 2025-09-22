# T1 gRPC Proto Generator

A C# source generator that creates clean SDK wrappers from .proto files for gRPC services. This tool simplifies gRPC client usage by generating strongly-typed wrapper classes that hide the complexity of gRPC channel management and provide a more intuitive API.

## Features

- 🚀 **Automatic Code Generation**: Generates client wrapper classes from .proto files at compile time
- 🎯 **Clean API**: Provides simplified, strongly-typed methods for gRPC service calls
- 🔧 **Channel Management**: Handles gRPC channel lifecycle automatically
- 📦 **Easy Integration**: Works as a NuGet package with zero configuration
- 🛡️ **Type Safety**: Full IntelliSense support and compile-time type checking
- ⚡ **Performance**: Minimal overhead with efficient code generation

## Installation

Install the NuGet package in your project:

```bash
dotnet add package T1.GrpcProtoGenerator
```

Or via Package Manager Console:

```powershell
Install-Package T1.GrpcProtoGenerator
```

## Usage

1. **Given a .proto file**:

```protobuf
syntax = "proto3";

service Greeter {
  rpc SayHello (HelloRequest) returns (HelloReply);
}

message HelloRequest {
  string name = 1;
}

message HelloReply {
  string message = 1;
}
```

2. **Add .proto files** to your project and ensure they are included as `<Protobuf>` items in your `.csproj`:

```xml
<ItemGroup>
    <!-- Keep standard Protobuf compilation for base types (both client and server) -->
    <Protobuf Include="Protos\greet.proto" GrpcServices="Both" ProtoRoot="Protos" />
    <Protobuf Include="Protos\Messages\requests.proto" GrpcServices="None" ProtoRoot="Protos" />
    <Protobuf Include="Protos\Messages\responses.proto" GrpcServices="None" ProtoRoot="Protos" />
    <AdditionalFiles Include="Protos\**\*.proto" />
</ItemGroup>

<ItemGroup>
<PackageReference Include="Grpc.AspNetCore" Version="2.64.0"/>
<PackageReference Include="Google.Api.CommonProtos" Version="2.15.0"/>
<PackageReference Include="Grpc.AspNetCore.Web" Version="2.64.0"/>
</ItemGroup>

<ItemGroup>
<Compile Remove="Generated\**" />
</ItemGroup>
```

2. **Build your project** - the source generator will automatically create wrapper classes for your gRPC services.

```csharp
public class GreeterService : IGreeterGrpcService
{
    public Task<HelloReplyGrpcDto> SayHello(HelloRequestGrpcDto request)
    {
        var response = new HelloReplyGrpcDto
        {
            Message = $"Hello {request.Name}!"
        };
        return Task.FromResult(response);
    }
}   
```

3. **Use the generated server wrappers**:

```csharp
builder.Services.AddGrpc();
builder.Services.AddScoped<IGreeterGrpcService, GreeterService>();

var app = builder.Build();
// Configure the HTTP request pipeline.
app.MapGrpcService<GreeterNativeGrpcService>();
```

4. **Use the generated client wrappers**:

```csharp
var services = new ServiceCollection();
// Configure gRPC server settings
services.Configure<GreeterGrpcConfig>(config =>
{
    config.ServerUrl = "https://localhost:7001"; // Your gRPC server address
});
// Register gRPC SDK using auto-generated extension method
services.AddGreeterGrpcSdk();
```

```csharp
// Generated wrapper provides a clean, easy-to-use API
var client = sp.GetRequiredService<IGreeterGrpcClient>();
var request = new HelloRequestGrpcDto 
{ 
    Name = "World from Consumer App" 
};
var response = await grpcClient.SayHelloAsync(request);
Console.WriteLine(response.Message);
```

## Generated Code Structure

For each gRPC service in your .proto files, the generator creates:

- **Server wrapper class**: `{ServiceName}GrpcService`
- **Client wrapper class**: `{ServiceName}GrpcClient`
- **Strongly-typed methods**: Async methods for each RPC call
- **Channel management**: Automatic connection handling
- **Error handling**: Proper exception propagation

## Configuration

The source generator works with standard gRPC tooling configuration. Make sure your .proto files are properly configured in your project file:

```xml
<ItemGroup>
  <Protobuf Include="Protos\**\*.proto" GrpcServices="Client" />
  <PackageReference Include="Grpc.AspNetCore" Version="2.57.0" />
  <PackageReference Include="T1.GrpcProtoGenerator" Version="1.0.0">
    <PrivateAssets>all</PrivateAssets>
    <IncludeAssets>runtime; build; native; contentfiles; analyzers</IncludeAssets>
  </PackageReference>
</ItemGroup>
```

## Requirements

- .NET Standard 2.1 or higher
- C# 8.0 or higher
- gRPC tooling (Grpc.Tools package)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/your-username/T1.GrpcProtoGenerator/issues) on GitHub.
