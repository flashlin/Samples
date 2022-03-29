using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class MergeExpr : SqlExpr
	{
		public string IntoToken { get; set; }
		public SqlExpr TargetTable { get; set; }
		public IdentExpr TargetAlias { get; set; }
		public SqlExpr SourceTable { get; set; }
		public IdentExpr SourceAlias { get; set; }
		public SqlExpr OnCondition { get; set; }
		public WhenMatchedExpr WhenMatched { get; set; }
		public WhenNotMatchedExpr WhenNotMatched { get; set; }
		public WhenNotMatchedExpr WhenNotMatchedBySource { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append("MERGE");
			if(!string.IsNullOrEmpty(IntoToken))
			{
				sb.Append(" INTO");
			}
			sb.Append($" {TargetTable}");
			if(TargetAlias != null)
			{
				sb.Append($" {TargetAlias}");
			}
			sb.AppendLine();
			sb.Append($"ON {SourceTable}");
			if(SourceAlias != null)
			{
				sb.Append($" {SourceAlias}");
			}
			sb.AppendLine();
			if(WhenMatched != null)
			{
				sb.AppendLine($"{WhenMatched}");
			}
			if(WhenNotMatched != null)
			{
				sb.AppendLine($"{WhenNotMatched}");
			}
			if(WhenNotMatchedBySource != null)
			{
				sb.AppendLine($"{WhenNotMatchedBySource}");
			}
			return sb.ToString();
		}
	}
}