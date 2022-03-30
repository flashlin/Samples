using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class IntoExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public SqlExprList Columns { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"INPUT {Table}");
			sb.Append("(");
			sb.Append($"{Columns}");
			sb.Append(")");
			return sb.ToString();
		}
	}
}