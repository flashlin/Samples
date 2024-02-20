using AspectCore.DynamicProxy;

namespace TestAspectCore;

public class LoggerAspect : AbstractInterceptorAttribute
{
    public override async Task Invoke(AspectContext context, AspectDelegate next)
    {
        try
        {
            // 在方法執行前添加日誌
            Console.WriteLine($"Calling method {context.ImplementationMethod.Name}");

            // 執行原始方法
            await next(context);

            // 在方法執行後添加日誌
            Console.WriteLine($"Method {context.ImplementationMethod.Name} executed successfully");
        }
        catch (Exception ex)
        {
            // 在方法執行過程中出現異常時添加日誌
            Console.WriteLine($"Method {context.ImplementationMethod.Name} threw an exception: {ex.Message}");
            throw;
        }
    }
}

public interface IMyClass
{
    void PublicMethod();
}

public class MyClass : IMyClass
{
    public void PublicMethod()
    {
        // 在這裡添加你的公共方法邏輯
        Console.WriteLine("Executing public method");
        PrivateMethod();
    }
    
    [LoggerAspect]
    private void PrivateMethod()
    {
        // 在這裡添加你的私有方法邏輯
        Console.WriteLine("Executing private method");
    }
}