namespace T1.ParserKit.ExprCollection;

public class SelectExpr : SqlExpr
{
    public List<SqlExpr> Columns { get; set; } = new();
    public SqlExpr? FromClause { get; set; }
}