using System.Text;
using T1.SqlDomParser;

namespace T1.SqlDom.Expressions
{
	public class SelectExpr : SqlExpr
	{
		public List<SqlExpr> Columns { get; set; } = new();
		public List<TableExpr> Tables { get; set; } = new();
		public SqlExpr WhereClause { get; set; }
		public override string ToSqlString()
		{
			var sb = new StringBuilder();
			sb.Append("SELECT ");
			sb.Append(Columns.ToSqlString(","));
			sb.Append("FROM ");
			sb.Append(Tables.ToSqlString(","));
			return sb.ToString();
		}
	}
}
