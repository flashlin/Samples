using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary
{
	public class InsertValuesExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public List<IdentExpr> Fields { get; set; }
		public List<List<SqlExpr>> ValuesList { get; set; }

		public override string ToString()
		{
			var fields = string.Join(",", Fields.Select(x => $"{x}"));
			var valuesList = string.Join(",", ValuesList.Select(x => $"({x})"));
			return $"INSERT {Table} ({fields}) VALUES {valuesList}";
		}
	}
}