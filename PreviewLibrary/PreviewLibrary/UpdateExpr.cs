using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class UpdateExpr : SqlExpr
	{
		public TopExpr Top { get; set; }
		public IdentExpr Table { get; set; }
		public WithOptionsExpr WithOptions { get; set; }
		public SqlExprList Fields { get; set; }
		public SqlExpr FromTableList { get; set; }
		public SqlExprList JoinTableList { get; set; }
		public SqlExpr WhereExpr { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"UPDATE");
			if (Top != null)
			{
				sb.Append($" {Top}");
			}
			sb.Append($" {Table}");
			if (WithOptions != null)
			{
				sb.Append($" {WithOptions}");
			}
			sb.AppendLine();
			sb.Append("SET ");
			foreach (var field in Fields.Items)
			{
				if (field != Fields.Items.First())
				{
					sb.Append(",");
				}
				sb.Append($"{field}");
			}

			if (FromTableList != null)
			{
				sb.AppendLine();
				sb.Append($"{FromTableList}");
			}

			if (JoinTableList.Items.Count > 0)
			{
				sb.AppendLine();
				sb.Append($"{JoinTableList}");
			}

			if (WhereExpr != null)
			{
				sb.AppendLine();
				sb.Append($"WHERE {WhereExpr}");
			}
			return sb.ToString();
		}
	}
}