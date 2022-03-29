using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;

namespace PreviewLibrary
{
	public class MergeInsertExpr : SqlExpr
	{
		public SqlExprList Fields { get; set; }
		public SqlExprList Values { get; set; }

		public override string ToString()
		{
			return $"INSERT ({Fields}) VALUES({Values})";
		}
	}
}