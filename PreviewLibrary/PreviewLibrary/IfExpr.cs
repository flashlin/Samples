using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary
{
	public class IfExpr : SqlExpr
	{
		public SqlExpr Condition { get; set; }
		public List<SqlExpr> Body { get; set; }
		public List<SqlExpr> ElseBody { get; set; }

		public override string ToString()
		{
			var body = string.Join("\r\n", Body.Select(x => $"{x}"));
			var elseBody = string.Empty;
			if( ElseBody.Count > 0 )
			{
				var content = string.Join("\r\n", ElseBody.Select(x => $"{x}"));
				elseBody = $"\r\nELSE BEGIN\r\n{content}\r\nEND";
			}
			return $"IF {Condition}\r\nBEGIN\r\n{body}\r\nEND{elseBody}";
		}
	}
}