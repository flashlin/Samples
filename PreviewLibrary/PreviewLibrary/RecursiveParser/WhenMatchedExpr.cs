using PreviewLibrary.Exceptions;
using System.Collections.Generic;

namespace PreviewLibrary
{
	public class WhenMatchedExpr : SqlExpr
	{
		public SqlExpr Condition { get; set; }
		public List<SqlExpr> Body { get; set; }

		public override string ToString()
		{
			return $"WHEN MATCH MATCHED {Condition} THEN\r\n{Body}";
		}
	}
}