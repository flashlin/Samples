namespace T1.SqlDom.Expressions;

public record ParseResult
{
    public static ParseResult Empty = new ParseResult();
    public SqlExpr Expr { get; set; } = SqlExpr.Empty;
    public bool Success { get; set; }
}