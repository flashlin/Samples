using System.Text.Json;
using System.Text.Json.Serialization;
using T1.Standard.Common;
using T1.Standard.DynamicCode;

namespace T1.LargeUtils;

public class LargeStreamProcessor
{
    public async Task<T> Read<T>(Stream stream)
        where T: class, new()
    {
        using var reader = new StreamReader(stream);
        using var jsonDocument = await JsonDocument.ParseAsync(stream);
        var rootElement = jsonDocument.RootElement;
        return (T)TraverseProperties(rootElement, typeof(T));
        // if (rootElement.TryGetProperty("items", out JsonElement itemsElement))
        // {
        //     if (itemsElement.ValueKind == JsonValueKind.Array)
        //     {
        //         foreach (JsonElement itemElement in itemsElement.EnumerateArray())
        //         {
        //             // 在這裡處理單個項目
        //             // 可以使用 itemElement.GetProperty("propertyName") 進行更進一步的屬性存取
        //
        //             // 例如：
        //             // if (itemElement.TryGetProperty("propertyName", out JsonElement propertyValue))
        //             // {
        //             //     // 在這裡處理特定屬性的值
        //             // }
        //         }
        //     }
        // }
    }

    private string GetJsonNameOfProperty(PropertyGetSetter prop)
    {
        var jsonNameAttr = (JsonPropertyNameAttribute)prop.Info.GetCustomAttributes(typeof(JsonPropertyNameAttribute), false)
            .FirstOrDefault(new JsonPropertyNameAttribute(prop.Name));
        return jsonNameAttr.Name;
    }

    private Dictionary<string, PropertyGetSetter> ReflectObjectProperties(Type objType)
    {
        var objInfo = ReflectionClass.Reflection(objType);
        var properties = new Dictionary<string, PropertyGetSetter>();
        foreach (var prop in objInfo.Properties)
        {
            var jsonName = GetJsonNameOfProperty(prop.Value);
            properties[jsonName] = prop.Value;
        }
        return properties;
    }

    private object TraverseProperties(JsonElement jsonElement, Type objType)
    {
        var obj = Activator.CreateInstance(objType)!;
        var properties = ReflectObjectProperties(objType);
        foreach (var property in jsonElement.EnumerateObject())
        {
            var jsonPropertyName = property.Name;
            var jsonPropertyValue = property.Value;
            if (!properties.TryGetValue(jsonPropertyName, out var prop))
            {
                continue;
            }
            
            if (jsonPropertyValue.ValueKind == JsonValueKind.Object)
            {
                var subValue = TraverseProperties(jsonPropertyValue, prop.PropertyType);
                prop.Setter(obj, subValue);
                continue;
            }

            if (jsonPropertyValue.ValueKind == JsonValueKind.Array)
            {
                if (prop.PropertyType == typeof(List<>))
                {
                    throw new Exception();
                }

                //var array = TraverseArray(jsonPropertyValue, itemType);
                continue;
            }

            var strValue = jsonPropertyValue.ToString();
            if (!string.IsNullOrEmpty(strValue))
            {
                prop.Setter(obj, strValue.ChangeType(prop.PropertyType)!);
            }
        }
        return obj;
    }
}