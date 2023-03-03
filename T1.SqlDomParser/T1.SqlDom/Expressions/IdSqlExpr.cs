namespace T1.SqlDom.Expressions;

public class IdSqlExpr : SqlExpr
{
    public override string ToSqlString()
    {
        return Value;
    }

    public string Value { get; set; } = string.Empty;
}