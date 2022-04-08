using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class ElseIfExpr : SqlExpr
	{
		public SqlExpr Condition { get; set; }
		public SqlExpr Body { get; set; }

		public override string ToString()
		{
			return $"ELSE IF {Condition}\r\n\t{Body}";
		}
	}
}