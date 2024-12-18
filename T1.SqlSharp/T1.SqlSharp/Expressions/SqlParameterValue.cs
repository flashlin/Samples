namespace T1.SqlSharp.Expressions;

public class SqlParameterValue : ISqlExpression
{
    public SqlType SqlType => SqlType.ParameterValue;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ParameterValue(this);
    }

    public string Name { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string ToSql()
    {
        return $@"{Name}={Value}";
    }
}