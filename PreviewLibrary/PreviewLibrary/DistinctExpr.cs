using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class DistinctExpr : SqlExpr
	{
		public SqlExpr RightSide { get; set; }

		public override string ToString()
		{
			return $"DISTINCT {RightSide}";
		}
	}
}