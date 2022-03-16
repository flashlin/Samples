using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class AssignSetExpr : SqlExpr
	{
		public IdentExpr Field { get; set; }
		public SqlExpr Value { get; set; }
	}
}