namespace T1.SqlSharp.Expressions;

public class SqlParameter : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ParameterValue;
    public TextSpan Span { get; set; } = new();

    public string ParameterName { get; set; } = string.Empty;
    public object? Value { get; set; }

    public void Accept(SqlVisitor visitor) { }

    public string ToSql()
    {
        return ParameterName;
    }
}
