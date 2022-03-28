using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class BetweenExpr : SqlExpr
	{
		public SqlExpr From { get; set; }
		public SqlExpr To { get; set; }

		public override string ToString()
		{
			return $"BETWEEN {From} AND {To}";
		}
	}
}