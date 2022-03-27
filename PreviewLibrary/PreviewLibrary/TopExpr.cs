using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class TopExpr : SqlExpr
	{
		public SqlExpr Count { get; set; }

		public override string ToString()
		{
			return $"TOP {Count}";
		}
	}
}