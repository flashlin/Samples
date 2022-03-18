using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class InsertFromSelectExpr : SqlExpr
	{
		public bool IntoToggle { get; set; }
		public IdentExpr Table { get; set; }
		public SelectExpr FromSelect { get; set; }
	}
}