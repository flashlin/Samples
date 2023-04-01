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
}