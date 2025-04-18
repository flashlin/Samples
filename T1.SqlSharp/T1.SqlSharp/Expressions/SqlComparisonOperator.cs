namespace T1.SqlSharp.Expressions;

public class SqlComparisonOperator : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ComparisonOperator;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ComparisonOperator(this);
    }

    public ComparisonOperator Value { get; set; }
    public string ToSql()
    {
        return Value.ToSql();
    }
}