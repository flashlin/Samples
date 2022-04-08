using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class CustomFuncExpr : SqlFuncExpr
	{
		public IdentExpr ObjectId { get; set; }
	}
}