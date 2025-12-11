namespace T1.SqlSharp.Helper;

public class SqlExpressionBuilderContext
{
    public string Schema { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public Dictionary<string, object?> Parameters { get; } = new();
}
