using PreviewLibrary.Exceptions;

namespace PreviewLibrary
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