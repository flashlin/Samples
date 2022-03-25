using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class UnionAllExpr : SqlExpr
	{
		public SqlExpr Next { get; set; }

		public override string ToString()
		{
			return $"UNION ALL\r\t{Next}";
		}
	}
}