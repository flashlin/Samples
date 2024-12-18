namespace T1.SqlSharp.Expressions;

public class SqlToken : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Token;
    public string Value { get; set; } = string.Empty;
    public required TextSpan Span { get; set; }
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SqlToken(this);
    }

    public string ToSql()
    {
        return Value;
    }

}