using System.Text;

namespace T1.SqlDom.Expressions;

public class ArithmeticExpr : SqlExpr
{
    public override string ToSqlString()
    {
        var sb = new StringBuilder();
        sb.Append(Left.ToSqlString());
        sb.Append(" " + Oper.ToSqlString() + " ");
        sb.Append(Right.ToSqlString());
        return sb.ToString();
    }

    public SqlExpr Left { get; set; } = Empty;
    public SqlExpr Oper { get; set; } = Empty;
    public SqlExpr Right { get; set; } = Empty;
}