using PreviewLibrary.Exceptions;

namespace PreviewLibrary.Expressions
{
	public class ArithmeticExpr : SqlExpr
	{
		public SqlExpr Left { get; set; }
		public string Oper { get; set; }
		public SqlExpr Right { get; set; }

		public override string ToString()
		{
			return $"{Left} {Oper} {Right}";
		}
	}
}