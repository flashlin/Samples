using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class AnyExpr : SqlExpr
	{
		public override string ToString()
		{
			return $"*";
		}
	}
}