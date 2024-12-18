namespace T1.SqlSharp.Expressions;

public class SqlNullValue : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Null;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_NullValue(this);
    }

    public string ToSql()
    {
        return "NULL";
    }
}