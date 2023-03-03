namespace T1.SqlDom.Expressions;

public class StringSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        return Value;
    }

    public string Value { get; init; } = string.Empty;
}