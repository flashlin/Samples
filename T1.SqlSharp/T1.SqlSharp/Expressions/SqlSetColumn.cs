namespace T1.SqlSharp.Expressions;

public class SqlSetColumn
{
    public string ColumnName { get; set; } = string.Empty;
    public string ParameterName { get; set; } = string.Empty;
    public object? Value { get; set; }
}

