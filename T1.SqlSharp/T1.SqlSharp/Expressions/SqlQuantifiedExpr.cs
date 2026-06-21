namespace T1.SqlSharp.Expressions;

public class SqlQuantifiedExpr : ISqlExpression
{
    public SqlType SqlType => SqlType.QuantifiedExpr;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_QuantifiedExpr(this);
    }

    public string Quantifier { get; set; } = string.Empty;
    public ISqlExpression? Subquery { get; set; }

    public string ToSql()
    {
        return $"{Quantifier} ({Subquery?.ToSql()})";
    }
}
