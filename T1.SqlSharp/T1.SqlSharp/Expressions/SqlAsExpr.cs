namespace T1.SqlSharp.Expressions;

public class SqlAsExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.AsExpr;
    public TextSpan Span { get; set; } = new();

    public required ISqlExpression Instance { get; set; }
    public required ISqlExpression As { get; set; }

    public string ToSql()
    {
        return $"{Instance.ToSql()} as {As.ToSql()}";
    }
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AsExpr(this);
    }
}