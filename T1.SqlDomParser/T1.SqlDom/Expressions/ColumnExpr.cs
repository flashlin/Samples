using System.Text;

namespace T1.SqlDom.Expressions;

public class ColumnExpr : SqlExpr
{
    public SqlExpr Name { get; set; } = SqlExpr.Empty;
    public SqlExpr Alias { get; set; } = SqlExpr.Empty;

    public override string ToSqlString()
    {
        var sb = new StringBuilder();
        sb.Append(Name.ToSqlString());
        sb.Append(Alias.ToSqlString());
        return sb.ToString();
    }
}