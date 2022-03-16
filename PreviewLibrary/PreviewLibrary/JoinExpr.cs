using PreviewLibrary.Exceptions;

namespace PreviewLibrary
{
	public class JoinExpr : SqlExpr
	{
		public TableExpr Table { get; set; }
		public SqlExpr Filter { get; set; }
		public JoinType JoinType { get; set; }

		public override string ToString()
		{
			return $"{JoinType} JOIN {Table} ON {Filter}";
		}
	}
}