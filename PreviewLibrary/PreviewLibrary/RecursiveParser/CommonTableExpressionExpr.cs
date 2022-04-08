using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class CommonTableExpressionExpr : SqlExpr
	{
		public SqlExpr InnerExpr { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"WITH");
			sb.Append($"{InnerExpr}");
			return sb.ToString();
		}
	}
}