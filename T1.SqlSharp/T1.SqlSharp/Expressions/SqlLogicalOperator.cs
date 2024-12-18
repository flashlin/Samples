namespace T1.SqlSharp.Expressions;

public class SqlLogicalOperator : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.LogicalOperator;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_LogicalOperator(this);
    }

    public LogicalOperator Value { get; set; }
    public string ToSql()
    {
        return Value.ToSql();
    }
}