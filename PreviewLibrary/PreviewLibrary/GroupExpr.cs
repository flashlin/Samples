using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class GroupExpr : SqlExpr
	{
		public SqlExpr Expr { get; set; }

		public override string ToString()
		{
			return $"({Expr})";
		}
	}
}