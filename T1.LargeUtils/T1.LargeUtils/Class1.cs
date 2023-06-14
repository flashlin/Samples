using System.Text.Json;
using System.Text.Json.Serialization;
using T1.Standard.Common;
using T1.Standard.DynamicCode;

namespace T1.LargeUtils;

public class LargeStreamProcessor
{
    public async Task Read<T>(Stream stream)
        where T: class, new()
    {
        var obj = new T();
        var classInfo = ReflectionClass.Reflection(typeof(T));
        
        using var reader = new StreamReader(stream);
        using var jsonDocument = await JsonDocument.ParseAsync(stream);
        var rootElement = jsonDocument.RootElement;

        foreach (var prop in classInfo.Properties)
        {
            var jsonNameAttr = (JsonPropertyNameAttribute)prop.Value.Info.GetCustomAttributes(typeof(JsonPropertyNameAttribute), false)
                .FirstOrDefault(new JsonPropertyNameAttribute(prop.Key));
            if (rootElement.TryGetProperty(jsonNameAttr.Name, out var jsonElement))
            {
                var text = jsonElement.GetString();
                if (text != null)
                {
                    var value = text.ChangeType(prop.Value.PropertyType)!;
                    prop.Value.Setter(obj, value);
                }
            }
        }
        
        
        if (rootElement.TryGetProperty("name", out JsonElement nameElement))
        {
            string name = nameElement.GetString();
            // 在這裡處理 name 屬性
        }

        if (rootElement.TryGetProperty("items", out JsonElement itemsElement))
        {
            if (itemsElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement itemElement in itemsElement.EnumerateArray())
                {
                    // 在這裡處理單個項目
                    // 可以使用 itemElement.GetProperty("propertyName") 進行更進一步的屬性存取

                    // 例如：
                    // if (itemElement.TryGetProperty("propertyName", out JsonElement propertyValue))
                    // {
                    //     // 在這裡處理特定屬性的值
                    // }
                }
            }
        }
    }
}