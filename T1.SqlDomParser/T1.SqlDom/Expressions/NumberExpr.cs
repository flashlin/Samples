namespace T1.SqlDom.Expressions;

public class NumberExpr : SqlExpr
{
    public string Value { get; set; } = string.Empty;
    public override string ToSqlString()
    {
        return Value;
    }
}