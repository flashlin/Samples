using AspectCore.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection;

namespace QueryKits.Extensions;

public static class StartupExtension
{
    public static void AddQueryKits(this IServiceCollection services)
    {
        services.ConfigureDynamicProxy();
    }

    // public static void UseQueryKits(IApplicationBuilder app)
    // {
    //     builder.Host.UseServiceProviderFactory(new DynamicProxyServiceProviderFactory());
    // }
}