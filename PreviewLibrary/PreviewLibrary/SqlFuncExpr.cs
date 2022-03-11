using System.Linq;

namespace PreviewLibrary
{
	public class SqlFuncExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr[] Arguments { get; set; }

		public override string ToString()
		{
			var args = string.Join(",", Arguments.Select(x => $"{x}"));
			return $"{Name}({args})";
		}
	}
}