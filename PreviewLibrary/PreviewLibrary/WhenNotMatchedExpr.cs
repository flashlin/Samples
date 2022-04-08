using PreviewLibrary.Exceptions;
using PreviewLibrary.Extensions;
using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary
{
	public class WhenNotMatchedExpr : SqlExpr
	{
		public SqlExpr Condition { get; set; }
		public string ByToken { get; set; }
		public List<SqlExpr> Body { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"WHEN NOT MATCHED");
			if(!string.IsNullOrEmpty(ByToken))
			{
				sb.Append($" {ByToken}");
			}
			sb.AppendLine($" {Condition}");
			sb.AppendLine("THEN");
			sb.Append($"{Body.MergeCodeLines()}");
			return sb.ToString();
		}
	}
}