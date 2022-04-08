using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class StringExpr : SqlExpr
	{
		public string Text { get; set; }

		public override string ToString()
		{
			return Text;
		}
	}
}