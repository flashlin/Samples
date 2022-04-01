using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class CommitExpr : SqlExpr
	{
		public override string ToString()
		{
			return $"COMMIT";
		}
	}
}