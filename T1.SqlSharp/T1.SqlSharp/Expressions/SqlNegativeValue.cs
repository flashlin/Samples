namespace T1.SqlSharp.Expressions;

public class SqlNegativeValue : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.NegativeValue;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_NegativeValue(this);
    }

    public required ISqlExpression Value { get; set; }

    public string ToSql()
    {
        return $"-{Value.ToSql()}";
    }
}