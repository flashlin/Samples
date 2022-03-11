using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class SelectExpr : SqlExpr
	{
		public List<SqlExpr> Fields { get; set; }
		public SqlExpr From { get; set; }
		public SqlExpr WhereExpr { get; set; }
		public List<JoinExpr> Joins { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			var fields = string.Join(",", Fields.Select(x => $"{x}"));
			sb.Append($"SELECT {fields}");
			if (From != null)
			{
				sb.Append($" FROM {From}");
			}
			if (Joins != null)
			{
				sb.Append(" " + string.Join("\r\n", Joins));
			}
			if (WhereExpr != null)
			{
				sb.Append($" {WhereExpr}");
			}
			return sb.ToString();
		}
	}
}