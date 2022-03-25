using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class CommonTableExpressionExpr : SqlExpr
	{
		public IdentExpr TableName { get; set; }
		public SqlExprList Columns { get; set; }
		public SqlExpr InnerExpr { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"WITH {TableName}");

			if (Columns != null && Columns.Items?.Count > 0)
			{
				sb.AppendLine($" ({Columns})");
			}

			sb.AppendLine($" AS (");
			sb.AppendLine($"\t{InnerExpr}");
			sb.Append($")");
			return sb.ToString();
		}
	}
}