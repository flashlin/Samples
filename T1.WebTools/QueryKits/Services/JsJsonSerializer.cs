using System.Text.Json;

namespace QueryKits.Services;

public class JsJsonSerializer : IJsJsonSerializer
{
    private readonly JsonSerializerOptions _options;

    public JsJsonSerializer()
    {
        _options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
    }
    
    public string Serialize(object obj)
    {
        return JsonSerializer.Serialize(obj, _options);
    }

    public T? Deserialize<T>(string json)
    {
        return JsonSerializer.Deserialize<T>(json);
    }
}