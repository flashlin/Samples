using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class AnyExpr : SqlExpr
	{
		public override string ToString()
		{
			return $"*";
		}
	}
}