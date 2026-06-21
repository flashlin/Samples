namespace T1.SqlSharp.Expressions;

public class SqlDefaultValue : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.DefaultValue;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_DefaultValue(this);
    }

    public string ToSql()
    {
        return "DEFAULT";
    }
}
