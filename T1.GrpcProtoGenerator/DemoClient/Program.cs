// See https://aka.ms/new-console-template for more information

using DemoServer.Services;
using Microsoft.Extensions.DependencyInjection;

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