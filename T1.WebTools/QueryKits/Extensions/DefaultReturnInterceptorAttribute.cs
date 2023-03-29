using AspectCore.DynamicProxy;

namespace QueryKits.Extensions;

public class DefaultReturnInterceptorAttribute : AbstractInterceptorAttribute
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
        if (
            IsGenericType(type, typeof(IList<>)) ||
            IsGenericType(type, typeof(List<>)) )
        {
            var listType = typeof(List<>).MakeGenericType(type.GetGenericArguments());
            return Activator.CreateInstance(listType);
        }

        if (IsGenericType(type, typeof(IEnumerable<>)))
        {
            var listType = typeof(List<>).MakeGenericType(type.GetGenericArguments());
            return Activator.CreateInstance(listType);
        }

        return type.IsValueType ? Activator.CreateInstance(type) : null;
    }

    public static bool IsGenericType(Type type, Type interfaceType)
    {
        return type.IsGenericType && type.GetGenericTypeDefinition() == interfaceType;
    }
}