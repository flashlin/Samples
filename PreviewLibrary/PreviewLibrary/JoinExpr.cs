using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary
{
	public class JoinExpr : SqlExpr
	{
		public TableExpr Table { get; set; }
		public SqlExpr Filter { get; set; }
		public JoinType JoinType { get; set; }
		public string OuterToken { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{JoinType}");
			if(!string.IsNullOrEmpty(OuterToken))
			{
				sb.Append($" {OuterToken}");
			}
			sb.Append($" JOIN {Table} ON {Filter}");
			return sb.ToString();
		}
	}
}