using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class GroupExpr : SqlExpr
	{
		public SqlExpr InnerExpr { get; set; }

		public override string ToString()
		{
			return $"({InnerExpr})";
		}
	}
}