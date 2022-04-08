using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class NotExpr : SqlExpr
	{
		public SqlExpr Right { get; set; }

		public override string ToString()
		{
			return $"NOT {Right}";
		}
	}
}