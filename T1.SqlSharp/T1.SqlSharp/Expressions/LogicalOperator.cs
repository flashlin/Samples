namespace T1.SqlSharp.Expressions;

public enum LogicalOperator
{
    None,
    And,
    Or,
    Not
}

public class SqlLogicalOperator : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.LogicalOperator;
    public TextSpan Span { get; set; } = new();
    public LogicalOperator Value { get; set; }
    public string ToSql()
    {
        return Value.ToSql();
    }
}
