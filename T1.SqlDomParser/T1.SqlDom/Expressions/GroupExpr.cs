namespace T1.SqlDom.Expressions;

public class GroupExpr : SqlExpr
{
    public SqlExpr Expr { get; set; } = SqlExpr.Empty;

    public override string ToSqlString()
    {
        return Expr.ToSqlString();
    }
}