using AspectCore.DynamicProxy;
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

public class DefaultReturnTypeInterceptorAttribute : AbstractInterceptorAttribute
{
    public override Task Invoke(AspectContext context, AspectDelegate next)
    {
        var returnType = context.ServiceMethod.ReturnType;
        if (returnType == typeof(void))
        {
            return Task.CompletedTask;
        }
        context.ReturnValue = GetDefaultValue(returnType);
        return Task.CompletedTask;
    }

    public static object? GetDefaultValue(Type type)
    {
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IList<>) || type.GetGenericTypeDefinition() == typeof(List<>))
        {
            var listType = typeof(List<>).MakeGenericType(type.GetGenericArguments());
            return Activator.CreateInstance(listType);
        }
        return type.IsValueType ? Activator.CreateInstance(type) : null;
    }
}