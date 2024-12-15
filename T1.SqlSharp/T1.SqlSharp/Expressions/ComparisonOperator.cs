namespace T1.SqlSharp.Expressions;

public enum ComparisonOperator
{
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Like,
    In,
    Between,
    Is,
    IsNot
}

public class SqlComparisonOperator : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ComparisonOperator;
    public TextSpan Span { get; set; } = new();
    public ComparisonOperator Value { get; set; }
    public string ToSql()
    {
        return Value.ToSql();
    }
}