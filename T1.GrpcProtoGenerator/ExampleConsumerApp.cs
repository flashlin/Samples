using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using DemoServer.Services; // Auto-generated from MyProtoContracts

namespace ExampleConsumerApp;

/// <summary>
/// Example showing how to use DemoSDK in an external project
/// </summary>
public class Program
{
    public static async Task Main(string[] args)
    {
        // Setup dependency injection
        var services = new ServiceCollection();
        
        // Configure gRPC server settings
        services.Configure<GreeterGrpcConfig>(config =>
        {
            config.ServerUrl = "https://localhost:7001"; // Your gRPC server address
        });
        
        // Register gRPC SDK using auto-generated extension method
        services.AddGreeterGrpcSdk();
        
        // Build service provider
        var serviceProvider = services.BuildServiceProvider();
        
        try
        {
            // Use the gRPC client
            var grpcClient = serviceProvider.GetRequiredService<IGreeterGrpcClient>();
            
            var request = new HelloRequestGrpcDto 
            { 
                Name = "World from Consumer App" 
            };
            
            var response = await grpcClient.SayHelloAsync(request);
            
            Console.WriteLine($"gRPC Response: {response.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        finally
        {
            serviceProvider.Dispose();
        }
    }
}

/* 
To use this in your project:

1. Add project reference to DemoSDK:
   <ProjectReference Include="path/to/DemoSDK/DemoSDK.csproj" />

2. Add required packages:
   <PackageReference Include="Microsoft.Extensions.DependencyInjection" Version="6.0.2" />
   <PackageReference Include="Microsoft.Extensions.Options" Version="6.0.0" />

3. Use the auto-generated classes:
   - GreeterGrpcConfig: Configure server URL
   - GreeterGrpcExtension.AddGreeterGrpcSdk(): Register services
   - IGreeterGrpcClient: Interface for dependency injection
   - HelloRequestGrpcDto/HelloReplyGrpcDto: Request/Response DTOs
*/
