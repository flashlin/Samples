using Microsoft.Extensions.DependencyInjection;
using DemoServer.Services;
using Grpc.Net.Client;
using Microsoft.Extensions.Options;

namespace DemoSDK;

public class StartupExample
{
    public class GreeterGrpcConfig
    {
        public string ServerUrl { get; set; } = "https://localhost:7001";
    }
    
    public void AddGreeterGrpcSdk(IServiceCollection services)
    {
        // Register gRPC Channel
        services.AddSingleton(provider =>
        {
            var config = provider.GetRequiredService<IOptions<GreeterGrpcConfig>>();
            return GrpcChannel.ForAddress(config.Value.ServerUrl);
        });

        // Register the original gRPC Client generated from proto
        services.AddTransient<Greeter.GreeterClient>(provider =>
        {
            var channel = provider.GetRequiredService<GrpcChannel>();
            return new Greeter.GreeterClient(channel);
        });

        // Register the wrapper gRPC Client interface and implementation
        services.AddTransient<IGreeterGrpcClient, GreeterGrpcClient>();
    }
}