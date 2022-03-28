using PreviewLibrary.Exceptions;

namespace PreviewLibrary
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