using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NSubstitute;

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

    public List<Type> QueryExternalDependencyInterfaces(IServiceCollection services)
    {
        var externalDependencies = services
            .Where(descriptor => descriptor.ServiceType.IsInterface)
            .Select(descriptor => descriptor.ServiceType)
            .Distinct()
            .Where(type => !type.Name.EndsWith("Service"))
            .ToList();
        return externalDependencies;
    }

    public List<object> MockInterfaces(IServiceCollection services, List<Type> interfaces)
    {
        var mockInstances = new List<object>();
        foreach (var interfaceType in interfaces)
        {
            var mockInstance = Substitute.For([interfaceType], []);
            mockInstances.Add(mockInstance);
            services.ReplaceWithMock(interfaceType, mockInstance);
        }
        return mockInstances;
    }
}