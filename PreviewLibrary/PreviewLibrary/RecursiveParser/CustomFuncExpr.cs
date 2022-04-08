using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class CustomFuncExpr : SqlFuncExpr
	{
		public IdentExpr ObjectId { get; set; }
	}
}