using System.Text;

namespace T1.SqlDom.Expressions;

public class TableExpr : SqlExpr
{
    public SqlExpr Name { get; set; } = SqlExpr.Empty;
    public SqlExpr Alias { get; set; } = SqlExpr.Empty;
    public bool IsSubQuery { get; set; }
    public override string ToSqlString()
    {
        var sb = new StringBuilder();
        return sb.ToString();
    }
}