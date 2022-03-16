using PreviewLibrary.Exceptions;

namespace PreviewLibrary.Expressions
{
	public class AssignSetExpr : SqlExpr
	{
		public IdentExpr Field { get; set; }
		public SqlExpr Value { get; set; }

		public override string ToString()
		{
			return $"{Field} = {Value}";
		}
	}
}