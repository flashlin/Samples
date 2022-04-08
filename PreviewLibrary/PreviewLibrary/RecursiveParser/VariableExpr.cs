using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class VariableExpr : SqlExpr
	{
		public string Name { get; set; }

		public override string ToString()
		{
			return $"{Name}";
		}
	}
}