using System.Text.Json;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Extensions;

public static class SqlJsonSerializerExtensions
{
    public static string ToSqlJsonString(this ISqlExpression sqlExpression)
    {
        var options = new JsonSerializerOptions().WithSqlExpressionConverter();
        return JsonSerializer.Serialize(sqlExpression, options);
    }
}