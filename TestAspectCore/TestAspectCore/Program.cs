// See https://aka.ms/new-console-template for more information

using AspectCore.Configuration;
using AspectCore.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection;
using TestAspectCore;

Console.WriteLine("Hello, AOP World!");

var services = new ServiceCollection();
services.ConfigureDynamicProxy(config => config.Interceptors.AddTyped<LoggerAspect>())
    .BuildDynamicProxyProvider();
services.AddTransient<IMyClass, MyClass>();
    
var serviceProvider = services.BuildServiceProvider();

var myClass = serviceProvider.GetRequiredService<IMyClass>();
myClass.PublicMethod();
Console.ReadKey();
Console.WriteLine("=== END ===");