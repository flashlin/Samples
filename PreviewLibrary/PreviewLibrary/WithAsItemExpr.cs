using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class WithAsItemExpr : SqlExpr
	{
		public IdentExpr AliasName { get; set; }
		public SqlExpr InnerSide { get; set; }

		public override string ToString()
		{
			return $"{AliasName} AS ( {InnerSide} )";
		}
	}
}