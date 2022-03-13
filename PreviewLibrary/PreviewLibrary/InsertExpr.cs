using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary
{
	public class InsertExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public List<IdentExpr> Fields { get; set; }
		public List<List<SqlExpr>> ValuesList { get; set; }
		public bool IntoToggle { get; set; }

		public override string ToString()
		{
			var intoToken = IntoToggle ? "INTO" : "";
			var fields = string.Join(",", Fields.Select(x => $"{x}"));
			var valuesList = string.Join(",", ValuesList.Select(x => $"({x})"));
			return $"INSERT {intoToken} {Table} ({fields}) VALUES {valuesList}";
		}
	}
}