using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class RankOverExpr : SqlExpr
	{
		public SqlExpr PartitionBy { get; set; }
		public string PartitionDescending { get; set; }
		public SqlExprList OrderByList { get; set; }
		public IdentExpr AliasName { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine("RANK() OVER(");
			if (PartitionBy != null)
			{
				sb.Append($"{PartitionBy}");
				if (string.IsNullOrEmpty(PartitionDescending))
				{
					sb.AppendLine($" {PartitionDescending}");
				}
			}
			sb.Append($"ORDER BY {OrderByList}");
			sb.AppendLine();
			sb.Append(")");
			if (AliasName != null)
			{
				sb.Append($" AS {AliasName}");
			}
			return sb.ToString();
		}
	}
}