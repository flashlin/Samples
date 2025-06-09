namespace T1.SqlSharp.Expressions;

public class SqlNotExpression : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.NotExpression;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_NotExpression(this);
    }

    public required ISqlExpression Value { get; set; }
    public string ToSql()
    {
        return $"NOT {Value.ToSql()}";
    }
}