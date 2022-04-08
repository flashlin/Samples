using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class IntoNewTableExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }

		public override string ToString()
		{
			return $"INTO {Table}";
		}
	}
}