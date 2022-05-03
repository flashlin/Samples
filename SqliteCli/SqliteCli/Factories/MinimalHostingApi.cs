using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace SqliteCli.Factories;

public class MinimalHostingApi : IHostBuilder
{
    private WebApplicationBuilder _webAppBuilder;
    private WebApplication _webApp;

    private MinimalHostingApi()
    {
        Properties = new Dictionary<object, object>();
    }

    public IWebHostEnvironment Environment
    {
        get
        {
            return _webAppBuilder.Environment;
        }
    }

    public static MinimalHostingApi CreateBuilder(string[] args)
    {
        return new MinimalHostingApi()
        {
            _webAppBuilder = WebApplication.CreateBuilder(args),
        };
    }

    public IHost Build()
    {
        _webApp = _webAppBuilder.Build();
        return _webApp;
    }

    public IHostBuilder ConfigureAppConfiguration(Action<HostBuilderContext, IConfigurationBuilder> configureDelegate)
    {
        throw new NotImplementedException();
    }

    public IHostBuilder ConfigureContainer<TContainerBuilder>(
        Action<HostBuilderContext, TContainerBuilder> configureDelegate)
    {
        throw new NotImplementedException();
    }

    public IHostBuilder ConfigureHostConfiguration(Action<IConfigurationBuilder> configureDelegate)
    {
        throw new NotImplementedException();
    }

    public IHostBuilder ConfigureServices(Action<HostBuilderContext, IServiceCollection> configureDelegate)
    {
        configureDelegate(null, _webAppBuilder.Services);
        return this;
    }

    public IHostBuilder UseServiceProviderFactory<TContainerBuilder>(IServiceProviderFactory<TContainerBuilder> factory)
    {
        throw new NotImplementedException();
    }

    public IHostBuilder UseServiceProviderFactory<TContainerBuilder>(
        Func<HostBuilderContext, IServiceProviderFactory<TContainerBuilder>> factory)
    {
        throw new NotImplementedException();
    }

    public IDictionary<object, object> Properties { get; }
}