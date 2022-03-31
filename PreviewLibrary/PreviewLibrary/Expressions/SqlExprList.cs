using PreviewLibrary.Exceptions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Expressions
{
	public class SqlExprList : SqlExpr
	{
		public List<SqlExpr> Items { get; set; }

		public bool HasComma { get; set; } = true;

		public bool IsEmpty()
		{
			return Items.Count == 0;
		}

		public override string ToString()
		{
			if (Items == null)
			{
				return string.Empty;
			}

			var comma = HasComma ? "," : "\r\n";
			return string.Join(comma, Items.Select(x => $"{x}"));
		}
	}
}