using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class TvpTableTypeExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }

		public override string ToString()
		{
			return $"{Name} READONLY";
		}
	}
}