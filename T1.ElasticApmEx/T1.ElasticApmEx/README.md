# T1.ElasticApmEx

## ElasticApmAspectAttribute

This attribute allows you to automatically instrument methods with Elastic APM. 

### Usage

1.  **Install the NuGet package:**

    ```bash
    dotnet add package T1.ElasticApmEx
    ```

2.  **Apply the `ElasticApmAspectAttribute` to your methods:**

    Decorate any method you want to trace with Elastic APM using the `[ElasticApmAspect]` attribute.

    ```csharp
    using T1.ElasticApmEx;
    using Ela.Apm;

    public class MyService
    {
        [ElasticApmAspect("CustomTransaction", "custom.type", "custom.subtype", "custom.action")]
        public void DoSomething()
        {
            // Your code here
            Tracer.CaptureSpan("MySpan", (string)null, (string)null, (Action)(() => {
                // Span work
                 Console.WriteLine("Executing DoSomething...");
            }));
        }

        [ElasticApmAspect("AnotherTransaction")]
        public async Task<string> DoSomethingAsync()
        {
            // Your asynchronous code here
            await Task.Delay(100);
            Console.WriteLine("Executing DoSomethingAsync...");
            return "Done";
        }
    }
    ```

3.  **Initialize Elastic APM in your application startup.**

    Ensure that Elastic APM is properly initialized in your `Program.cs` or `Startup.cs` file. For .NET Core applications, you typically add `app.UseElasticApm()`.

    ```csharp
    // In Program.cs
    var builder = WebApplication.CreateBuilder(args);
    builder.Services.AddControllers();
    // ... other services

    var app = builder.Build();

    app.UseElasticApm(); // Add this line to enable Elastic APM
    
    // ... other middleware

    app.Run();
    ```

### Attribute Parameters

*   `transactionName` (string): The name of the APM transaction. If not provided, the method name will be used.
*   `spanType` (string): The type of the APM span. Default is `"custom"`.
*   `spanSubtype` (string): The subtype of the APM span. Default is `"method"`.
*   `spanAction` (string): The action of the APM span. Default is `"execute"`.

### Example with Parameters

```csharp
[ElasticApmAspect("UserService.GetUser", "db", "postgresql", "query")]
public User GetUserById(int userId)
{
    // ... database query logic
    return new User { Id = userId, Name = "Test User" };
}
```