using PreviewLibrary.Exceptions;

namespace PreviewLibrary.RecursiveParser
{
	public class ColumnDescendingExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public string Descending { get; set; }

		public override string ToString()
		{
			if (string.IsNullOrEmpty(Descending))
			{
				return $"{Name}";
			}
			return $"{Name} {Descending}";
		}
	}
}