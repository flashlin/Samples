using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class SqlVariableExpr : SqlExpr
	{
		public string Name { get; set; }

		public override string ToString()
		{
			return $"{Name}";
		}
	}
}