using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class InExpr : SqlExpr
	{
		public SqlExpr LeftExpr { get; set; }
		public SqlExpr Values { get; set; }

		public override string ToString()
		{
			return $"{LeftExpr} IN ({Values})";
		}
	}
}