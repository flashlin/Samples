using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class CommonTableExpressionExpr : SqlExpr
	{
		public IdentExpr TableName { get; set; }
		public SqlExprList Columns { get; set; }
		public SelectExpr FirstSelect { get; set; }
		public SelectExpr RecursiveSelect { get; set; }
		public SelectExpr Query { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"WITH {TableName} ({Columns})");
			sb.AppendLine($"AS (");
			sb.AppendLine("\t" + FirstSelect.ToString());
			sb.AppendLine("\tUNION ALL");
			sb.AppendLine("\t" + RecursiveSelect.ToString());
			sb.AppendLine($")");
			sb.Append(Query.ToString());
			return sb.ToString();
		}
	}
}