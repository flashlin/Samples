using System;
using System.Reflection;
using T1.GrpcSourceGenerator.Demo;

// 顯示所有生成的類型
Console.WriteLine("Generated types in the assembly:");
foreach (Type type in typeof(Program).Assembly.GetTypes())
{
    Console.WriteLine($"- {type.FullName}");
}

// 顯示帶有特定後綴的類型
Console.WriteLine("\nGreater related types:");
foreach (Type type in typeof(Program).Assembly.GetTypes())
{
    if (type.Name.Contains("Greeter") || type.Name.EndsWith("Message"))
    {
        Console.WriteLine($"- {type.FullName}");
        
        // 顯示類型的成員
        Console.WriteLine("  Properties:");
        foreach (PropertyInfo prop in type.GetProperties())
        {
            Console.WriteLine($"    {prop.PropertyType.Name} {prop.Name}");
        }
        
        // 顯示方法
        Console.WriteLine("  Methods:");
        foreach (MethodInfo method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (method.IsSpecialName) continue; // 跳過屬性方法
            Console.WriteLine($"    {method.ReturnType.Name} {method.Name}({string.Join(", ", method.GetParameters().Select(p => $"{p.ParameterType.Name} {p.Name}"))})");
        }
        
        Console.WriteLine();
    }
}
