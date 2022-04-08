using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;

namespace PreviewLibrary.RecursiveParser
{
	public class GroupByExpr : SqlExpr
	{
		public SqlExprList Items { get; set; }

		public override string ToString()
		{
			return $"GROUP BY {Items}";
		}
	}
}