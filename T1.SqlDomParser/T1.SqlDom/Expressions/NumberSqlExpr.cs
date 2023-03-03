namespace T1.SqlDom.Expressions;

public class NumberSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        return Value;
    }

    public string Value { get; init; } = null!;
}