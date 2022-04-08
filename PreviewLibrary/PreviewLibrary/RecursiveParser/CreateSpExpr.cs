using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using PreviewLibrary.Extensions;
using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary
{
	public class CreateSpExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public SqlExprList Arguments { get; set; }
		public List<SqlExpr> Body { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"CREATE PROCEDURE {Name}");
			sb.Append($"{Arguments}");
			sb.AppendLine($"\r\nAS");
			sb.AppendLine($"BEGIN");
			sb.AppendLine($"{Body.MergeCodeLines()}");
			sb.Append($"END");
			return sb.ToString();
		}
	}
}