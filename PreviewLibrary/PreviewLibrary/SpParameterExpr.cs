using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class SpParameterExpr : SqlExpr
	{
		public string Name { get; set; }
		public SqlExpr Value { get; set; }
	}
}