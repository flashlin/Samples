namespace T1.SqlDom.Expressions;

public class StringExpr : SqlExpr
{
    public string Value { get; set; } = string.Empty;

    public override string ToSqlString()
    {
        return Value;
    }

    public override string ToString()
    {
        return ToSqlString();
    }
}