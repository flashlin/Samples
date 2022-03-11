using System.Linq;

namespace PreviewLibrary
{
	public class ExecuteExpr : SqlExpr
	{
		public IdentExpr Method { get; set; }
		public SqlExpr[] Arguments { get; set; }

		public override string ToString()
		{
			var args = string.Join(",", Arguments.Select(x => $"{x}"));
			return $"EXEC {Method} {args}";
		}
	}
}