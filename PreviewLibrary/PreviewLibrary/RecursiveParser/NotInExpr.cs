using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class NotInExpr : SqlExpr
	{
		public SqlExpr Left { get; set; }
		public SqlExprList Right { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{Left} NOT IN");
			sb.Append("(");
			sb.Append($" {Right} ");
			sb.Append(")");
			return sb.ToString();
		}
	}
}