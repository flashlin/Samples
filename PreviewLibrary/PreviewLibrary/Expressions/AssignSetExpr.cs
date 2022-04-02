using PreviewLibrary.Exceptions;

namespace PreviewLibrary.Expressions
{
	public class AssignSetExpr : SqlExpr
	{
		public SqlExpr Field { get; set; }
		public SqlExpr Value { get; set; }

		public override string ToString()
		{
			return $"{Field} {Value}";
		}
	}
}