namespace T1.SqlSharp.Expressions;

public class SqlAtTimeZoneExpr : ISqlExpression
{
    public SqlType SqlType => SqlType.AtTimeZoneExpr;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AtTimeZoneExpr(this);
    }

    public required ISqlExpression Expression { get; set; }
    public required ISqlExpression TimeZone { get; set; }

    public string ToSql()
    {
        return $"{Expression.ToSql()} AT TIME ZONE {TimeZone.ToSql()}";
    }
}
