using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class NegativeExpr : SqlExpr
	{
		public SqlExpr Value { get; set; }

		public override string ToString()
		{
			return $"-{Value}";
		}
	}
}