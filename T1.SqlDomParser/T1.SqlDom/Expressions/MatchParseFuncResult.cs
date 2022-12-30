namespace T1.SqlDom.Expressions;

public class MatchParseFuncResult
{
    public ParseFunc? Func { get; set; }
    public SqlExpr Expr { get; set; } = null!;
}