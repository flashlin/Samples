using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary
{
	public class WhileExpr : SqlExpr
	{
		public SqlExpr Condition { get; set; }
		public List<SqlExpr> Body { get; set; }

		public override string ToString()
		{
			var body = string.Join("\r\n", Body.Select(x => $"{x}"));
			return $"WHILE {Condition}\r\nBEGIN\r\n{body}\r\nEND";
		}
	}
}