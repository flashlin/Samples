using AspectCore.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection;
using QueryKits.Services;

namespace QueryKits.Extensions;

public static class StartupExtension
{
    public static void AddQueryKits(this IServiceCollection services)
    {
        services.ConfigureDynamicProxy();
        services.AddTransient<IDbContextOptionsFactory, DbContextOptionsFactory>();
    }

    // public static void UseQueryKits(IApplicationBuilder app)
    // {
    //     builder.Host.UseServiceProviderFactory(new DynamicProxyServiceProviderFactory());
    // }
    public static IServiceCollection AddEventAggregator(this IServiceCollection services, Action<EventAggregatorOptions>? configure = null)
    {
        services.AddSingleton<IEventAggregator, EventAggregator>();
        if (configure != null)
        {
            services.Configure(configure);
        }
        return services;
    }
}
