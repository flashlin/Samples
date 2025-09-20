using Microsoft.Extensions.DependencyInjection;
using DemoServer.Services;
using Grpc.Net.Client;

namespace DemoSDK;

public class StartupExample
{
    public void AddSdk(IServiceCollection services, string grpcServerAddress = "https://localhost:7001")
    {
        // Register gRPC Channel
        services.AddSingleton(provider => GrpcChannel.ForAddress(grpcServerAddress));

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