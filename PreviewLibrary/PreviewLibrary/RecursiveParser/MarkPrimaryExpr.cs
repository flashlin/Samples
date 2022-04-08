using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class MarkExpr : SqlExpr
	{
		public string Token { get; set; }

		public override string ToString()
		{
			return $"{Token}";
		}
	}
}