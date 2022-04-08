using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class AssignExpr : SqlExpr
	{
		public object Value { get; set; }

		public override string ToString()
		{
			return $"= {Value}";
		}
	}
}