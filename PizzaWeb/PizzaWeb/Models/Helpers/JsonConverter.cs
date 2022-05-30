using System.Text.Json;
using Microsoft.EntityFrameworkCore.ValueGeneration;

namespace PizzaWeb.Models.Helpers;

public class JsonConverter : IJsonConverter
{
    public T Deserialize<T>(string variablesData)
    {
        var jsonOptions = new JsonSerializerOptions
        {
            AllowTrailingCommas = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
        return JsonSerializer.Deserialize<T>(variablesData, jsonOptions)!;
    }

    public string Serialize<T>(T data)
    {
        var jsonOptions = new JsonSerializerOptions
        {
            AllowTrailingCommas = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
        return JsonSerializer.Serialize(data, jsonOptions);
    }
}

public interface IServiceLocator
{
    T GetService<T>();
}

public class ServiceLocator : IServiceLocator
{
    private IServiceProvider? _currentServiceProvider;
    private static IServiceProvider? _serviceProvider;
    
    public ServiceLocator(IServiceProvider? currentServiceProvider)
    {
        _currentServiceProvider = currentServiceProvider;
    }

    public T GetService<T>()
    {
        return _serviceProvider!.GetService<T>()!;
    }
    
    public static void SetLocatorProvider(IServiceProvider? serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }
    
    public static IServiceLocator Current
    {
        get
        {
            return new ServiceLocator(_serviceProvider);
        }
    }
}