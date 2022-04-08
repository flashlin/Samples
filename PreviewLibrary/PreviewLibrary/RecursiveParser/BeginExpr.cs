using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class BeginExpr : SqlExpr
	{
		public List<SqlExpr> Body { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine("BEGIN");
			var body = string.Join("\r\n", Body.Select(x => $"{x}"));
			sb.AppendLine(body);
			sb.Append("END");
			return sb.ToString();
		}
	}
}