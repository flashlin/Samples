namespace T1.SqlDom.Expressions;

public record ParseResult
{
    public static readonly ParseResult Empty = new ParseResult();
    public static ParseResult Ok(SqlExpr expr)
    {
        return new ParseResult
        {
            Expr = expr,
            Success = true,
        };
    }
    public static ParseResult Fail(string expected)
    {
        return new ParseResult
        {
            Expected = expected,
            Success = false
        };
    }
    public SqlExpr Expr { get; set; } = SqlExpr.Empty;
    public string Expected { get; set; } = string.Empty;
    public bool Success { get; set; }
}