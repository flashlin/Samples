using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class JoinAllExpr : SqlExpr
	{
		public SqlExpr Next { get; set; }

		public override string ToString()
		{
			return $"JOIN ALL\r\n{Next}";
		}
	}
}