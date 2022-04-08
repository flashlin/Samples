using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class NullExpr : SqlExpr
	{
		public string Token { get; set; }

		public override string ToString()
		{
			return "NULL";
		}
	}
}