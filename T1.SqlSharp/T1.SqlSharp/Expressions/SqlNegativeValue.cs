namespace T1.SqlSharp.Expressions;

public class SqlNegativeValue : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.NegativeValue;
    public TextSpan Span { get; set; } = new();
    public required ISqlExpression Value { get; set; }

    public string ToSql()
    {
        return $"-{Value.ToSql()}";
    }
}

public class SqlNotExpression : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.NotExpression;
    public TextSpan Span { get; set; } = new();
    public required ISqlExpression Value { get; set; }
    public string ToSql()
    {
        return $"NOT {Value.ToSql()}";
    }
}

public class SqlExistsExpression : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.ExistsExpression;
    public TextSpan Span { get; set; } = new();
    public required ISqlExpression Query { get; set; }
    public string ToSql()
    {
        return $"EXISTS ({Query.ToSql()})";
    }
}