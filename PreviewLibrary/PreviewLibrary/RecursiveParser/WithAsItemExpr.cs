using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class WithAsItemExpr : SqlExpr
	{
		public IdentExpr TableName { get; set; }
		public SqlExprList Columns { get; set; }
		public IdentExpr AliasName { get; set; }
		public SqlExpr InnerSide { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			if (TableName != null)
			{
				sb.Append($"{TableName}");
			}
			if (Columns != null && !Columns.IsEmpty())
			{
				sb.Append($" ({Columns})");
			}
			if (AliasName != null)
			{
				sb.Append($" {AliasName}");
			}
			sb.AppendLine();
			sb.AppendLine($"AS (");
			sb.AppendLine($"{InnerSide}");
			sb.Append(")");
			return sb.ToString();
		}
	}
}