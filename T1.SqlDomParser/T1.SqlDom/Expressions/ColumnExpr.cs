using System.Text;

namespace T1.SqlDom.Expressions;

public class ColumnExpr : SqlExpr
{
    public SqlExpr Name { get; set; }
    public SqlExpr Alias { get; set; }

    public override string ToSqlString()
    {
        var sb = new StringBuilder();
        sb.Append(Name.ToSqlString());
        sb.Append(Alias.ToSqlString());
        return sb.ToString();
    }
}