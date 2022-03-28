using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class BetweenExpr : SqlExpr
	{
		public SqlExpr Left { get; set; }
		public SqlExpr From { get; set; }
		public SqlExpr To { get; set; }

		public override string ToString()
		{
			return $"{Left} BETWEEN {From} AND {To}";
		}
	}
}