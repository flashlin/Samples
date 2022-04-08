using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class MaxExpr : SqlExpr
	{
		public override string ToString()
		{
			return $"MAX";
		}
	}
}