using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.Expressions
{
	public class CaseExpr : SqlExpr
	{
		public List<WhenThenExpr> WhenList { get; set; }
		public SqlExpr Else { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine("CASE");
			foreach (var whenExpr in WhenList)
			{
				sb.AppendLine($"	{whenExpr}");
			}
			if (Else != null)
			{
				sb.AppendLine($"	ELSE {Else}");
			}
			sb.Append("END");
			return sb.ToString();
		}
	}
}