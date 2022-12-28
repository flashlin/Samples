using System.Text;

namespace T1.SqlDom.Expressions;

public class ComparsionExpr : SqlExpr
{
    public SqlExpr Left { get; set; } = SqlExpr.Empty;
    public SqlExpr Right { get; set; } = SqlExpr.Empty;
    public SqlExpr Oper { get; set; } = SqlExpr.Empty;

    public override string ToSqlString()
    {
        var sb = new StringBuilder();
        sb.Append(Left.ToSqlString());
        sb.Append(" " + Oper.ToSqlString() + " ");
        sb.Append(Right.ToSqlString());
        return sb.ToString();
    }
}