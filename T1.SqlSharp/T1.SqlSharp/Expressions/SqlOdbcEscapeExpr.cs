namespace T1.SqlSharp.Expressions;

public class SqlOdbcEscapeExpr : ISqlExpression
{
    public SqlType SqlType => SqlType.OdbcEscapeExpr;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_OdbcEscapeExpr(this);
    }

    public string Keyword { get; set; } = string.Empty;
    public ISqlExpression? Body { get; set; }

    public string ToSql()
    {
        return $"{{ {Keyword} {Body?.ToSql()} }}";
    }
}
