using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class SelectExpr : SqlExpr
	{
		public TopExpr TopExpr { get; set; }
		public SqlExprList Fields { get; set; }
		public SqlExpr From { get; set; }
		public SqlExprList FromJoinList { get; set; }
		public SqlExpr WhereExpr { get; set; }
		public List<SqlExpr> Joins { get; set; }
		public GroupByExpr GroupByExpr { get; set; }
		public List<SqlExpr> JoinAllList { get; set; }
		public SqlExprList OrderByExpr { get; set; }
		public IntoNewTableExpr IntoNewTable { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"SELECT");
			if (TopExpr != null)
			{
				sb.Append($" {TopExpr}");
			}
			sb.Append($" {Fields}");

			if (IntoNewTable != null)
			{
				sb.AppendLine();
				sb.Append($"{IntoNewTable}");
			}

			if (From != null)
			{
				sb.Append($" FROM {From}");
			}

			if (FromJoinList != null && FromJoinList.Items.Count > 0)
			{
				sb.AppendLine();
				sb.AppendLine($"{FromJoinList}");
			}

			if (Joins != null && Joins.Count > 0)
			{
				sb.Append(" " + string.Join("\r\n", Joins));
			}
			if (WhereExpr != null)
			{
				sb.Append($" WHERE {WhereExpr}");
			}
			if (GroupByExpr != null)
			{
				sb.Append($"\r\n\t{GroupByExpr}");
			}
			if (OrderByExpr != null)
			{
				sb.Append($"\r\n\tORDER BY {OrderByExpr}");
			}
			if (JoinAllList != null && JoinAllList.Count > 0)
			{
				sb.AppendLine();
				sb.Append(string.Join("\r\n", JoinAllList.Select(x => $"{x}")));
			}
			return sb.ToString();
		}
	}
}