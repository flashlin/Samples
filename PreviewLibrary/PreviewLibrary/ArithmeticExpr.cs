using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class ArithmeticExpr : SqlExpr
	{
		public SqlExpr Left { get; set; }
		public string Oper { get; set; }
		public SqlExpr Right { get; set; }
	}
}