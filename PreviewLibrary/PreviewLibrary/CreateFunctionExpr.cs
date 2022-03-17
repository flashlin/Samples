using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class CreateFunctionExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public SqlExprList Arguments { get; set; }
		public SqlExpr ReturnDataType { get; set; }
		public List<SqlExpr> Body { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"CREATE FUNCTION {Name}");
			sb.AppendLine($"(");
			sb.AppendLine($"{Arguments}");
			sb.AppendLine($")");
			sb.AppendLine($"RETURNS {ReturnDataType}");
			sb.AppendLine("AS BEGIN");
			sb.AppendLine(string.Join("\r\n", Body.Select(x => $"{x}")));
			sb.AppendLine("END");
			return sb.ToString();
		}
	}
}