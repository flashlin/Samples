using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;

namespace T1.IntegrationTesting;

public class TestWebFactory
{
    public IHost CreateGrpcServer<TGrpcServerStartup>(int port, Action<IServiceCollection> mockServices) 
        where TGrpcServerStartup : class
    {
        var server = Host.CreateDefaultBuilder()
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<TGrpcServerStartup>();
                webBuilder.UseUrls($"http://localhost:{port}");
                webBuilder.ConfigureServices(mockServices);
            })
            .Build();
        return server;
    }
}