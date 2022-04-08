using PreviewLibrary.Exceptions;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class ExecuteExpr : SqlExpr
	{
		public SqlExpr[] Arguments { get; set; }
		public string ExecName { get; set; }
		public string LeftSide { get; set; }
		public SqlExpr Method { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{ExecName.ToUpper()}");
			
			if (!string.IsNullOrEmpty(LeftSide))
			{
				sb.Append($" {LeftSide} =");
			}
			
			sb.Append($" {Method}");

			if (Arguments != null)
			{
				var args = string.Join(",", Arguments.Select(x => $"{x}"));
				sb.Append($" {args}");
			}
			return sb.ToString();
		}
	}
}