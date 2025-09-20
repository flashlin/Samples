using Microsoft.Extensions.DependencyInjection;
using DemoServer.Services;
using Grpc.Net.Client;
using Microsoft.Extensions.Options;
using System;

namespace DemoSDK;

public class GreeterGrpcConfig
{
    public string ServerUrl { get; set; } = "https://localhost:7001";
}

public static class GreeterGrpcExtension
{
    public static void AddGreeterGrpcSdk(this IServiceCollection services)
    {
        services.AddTransient<Greeter.GreeterClient>(provider =>
        {
            var config = provider.GetRequiredService<IOptions<GreeterGrpcConfig>>();
            var channel = GrpcChannel.ForAddress(config.Value.ServerUrl);
            return new Greeter.GreeterClient(channel);
        });
        services.AddTransient<IGreeterGrpcClient, GreeterGrpcClient>();
    }
}