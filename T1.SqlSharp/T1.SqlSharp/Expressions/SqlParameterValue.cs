namespace T1.SqlSharp.Expressions;

public class SqlParameterValue : ISqlExpression
{
    public SqlType SqlType => SqlType.ParameterValue;
    public TextSpan Span { get; set; } = new();
    public string Name { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string ToSql()
    {
        return $@"{Name}={Value}";
    }
}