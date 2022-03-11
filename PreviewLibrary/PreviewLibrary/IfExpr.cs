using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary
{
	public class IfExpr : SqlExpr
	{
		public SqlExpr Condition { get; set; }
		public List<SqlExpr> Body { get; set; }

		public override string ToString()
		{
			var body = string.Join("\r\n", Body.Select(x => $"{x}"));
			return $"IF {Condition} BEGIN {body} END";
		}
	}
}