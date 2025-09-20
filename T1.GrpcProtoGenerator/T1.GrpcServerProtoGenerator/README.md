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

1. **Add .proto files** to your project and ensure they are included as `<Protobuf>` items in your `.csproj`:

```xml
<ItemGroup>
  <Protobuf Include="Protos\greet.proto" GrpcServices="Client" />
</ItemGroup>
```

2. **Build your project** - the source generator will automatically create wrapper classes for your gRPC services.

3. **Use the generated client wrappers**:

```csharp
// Generated wrapper provides a clean, easy-to-use API
var client = new GreeterServiceClient("https://localhost:5001");
var response = await client.SayHelloAsync(new HelloRequest { Name = "World" });
Console.WriteLine(response.Message);
```

## Generated Code Structure

For each gRPC service in your .proto files, the generator creates:

- **Client wrapper class**: `{ServiceName}Client`
- **Strongly-typed methods**: Async methods for each RPC call
- **Channel management**: Automatic connection handling
- **Error handling**: Proper exception propagation

## Example

Given a .proto file:

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

The generator creates:

```csharp
public partial class GreeterServiceClient
{
    private readonly GrpcChannel _channel;
    private readonly Greeter.GreeterClient _client;

    public GreeterServiceClient(string address)
    {
        _channel = GrpcChannel.ForAddress(address);
        _client = new Greeter.GreeterClient(_channel);
    }

    public async Task<HelloReply> SayHelloAsync(HelloRequest request)
    {
        return await _client.SayHelloAsync(request);
    }
}
```

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
