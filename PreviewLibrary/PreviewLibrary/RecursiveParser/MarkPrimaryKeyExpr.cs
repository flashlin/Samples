using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class MarkPrimaryKeyExpr : SqlExpr
	{
		public override string ToString()
		{
			return "PRIMARY KEY";
		}
	}
}