using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class MaxExpr : SqlExpr
	{
		public override string ToString()
		{
			return $"MAX";
		}
	}
}