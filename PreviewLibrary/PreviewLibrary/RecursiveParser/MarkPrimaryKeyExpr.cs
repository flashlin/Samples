using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class MarkPrimaryKeyExpr : SqlExpr
	{
		public override string ToString()
		{
			return "PRIMARY KEY";
		}
	}
}