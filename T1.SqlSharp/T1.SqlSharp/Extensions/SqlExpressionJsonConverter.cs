using System.Collections;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Extensions;

public class SqlExpressionJsonConverter : JsonConverter<object>
{
    public override bool CanConvert(Type objectType)
    {
        return typeof(ISqlExpression).IsAssignableFrom(objectType) ||
               (objectType.IsGenericType &&
                (objectType.GetGenericTypeDefinition() == typeof(List<>) ||
                 objectType.GetGenericTypeDefinition() == typeof(IEnumerable<>)) &&
                typeof(ISqlExpression).IsAssignableFrom(objectType.GetGenericArguments()[0]));
    }

    public override object Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        throw new NotImplementedException("Deserialization is not supported.");
    }

    public override void Write(Utf8JsonWriter writer, object? value, JsonSerializerOptions options)
    {
        if (value == null)
        {
            writer.WriteNullValue();
            return;
        }

        if (value is IEnumerable enumerable && !(value is string))
        {
            writer.WriteStartArray();
            foreach (var item in enumerable)
            {
                WriteFullObject(writer, item, options);
            }

            writer.WriteEndArray();
            return;
        }

        WriteFullObject(writer, value, options);
    }

    private void WriteFullObject(Utf8JsonWriter writer, object? obj, JsonSerializerOptions options)
    {
        if (obj == null || !(obj is ISqlExpression))
        {
            JsonSerializer.Serialize(writer, obj, obj?.GetType() ?? typeof(object), options);
            return;
        }

        writer.WriteStartObject();
        var properties = obj.GetType()
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.CanRead);
        foreach (var property in properties)
        {
            writer.WritePropertyName(property.Name);
            var propertyValue = property.GetValue(obj);
            if (propertyValue == null)
            {
                writer.WriteNullValue();
                continue;
            }

            if (propertyValue is ISqlExpression sqlExpression)
            {
                WriteFullObject(writer, sqlExpression, options);
                continue;
            }

            if (propertyValue is IEnumerable enumerable && !(propertyValue is string))
            {
                var array = enumerable.Cast<object>().ToArray();
                if (array.Any(item => item is ISqlExpression))
                {
                    writer.WriteStartArray();
                    foreach (var item in array)
                    {
                        if (item is ISqlExpression sqlExpressionItem)
                        {
                            WriteFullObject(writer, sqlExpressionItem, options);
                        }
                        else
                        {
                            JsonSerializer.Serialize(writer, item, item?.GetType() ?? typeof(object), options);
                        }
                    }

                    writer.WriteEndArray();
                    continue;
                }
            }

            JsonSerializer.Serialize(writer, propertyValue, property.PropertyType, options);
        }

        writer.WriteEndObject();
    }
}