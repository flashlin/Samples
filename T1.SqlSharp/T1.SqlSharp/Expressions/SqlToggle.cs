namespace T1.SqlSharp.Expressions;

public class SqlToggle : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WithToggle;
    public string ToggleName { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;


    public string ToSql()
    {
        return $"{ToggleName}={Value}";
    }
}