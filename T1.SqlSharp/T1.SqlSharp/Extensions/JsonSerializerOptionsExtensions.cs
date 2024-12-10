using System.Text.Json;
using System.Text.Json.Serialization;

namespace T1.SqlSharp.Extensions;

public static class JsonSerializerOptionsExtensions
{
    public static JsonSerializerOptions WithSqlExpressionConverter(
        this JsonSerializerOptions options)
    {
        options.Converters.Add(new SqlExpressionJsonConverter());
        options.Converters.Add(new JsonStringEnumConverter());
        options.WriteIndented = true;
        options.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
        return options;
    }
}