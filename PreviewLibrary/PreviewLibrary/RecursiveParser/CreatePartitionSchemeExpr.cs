using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class CreatePartitionSchemeExpr : SqlExpr
	{
		public IdentExpr SchemeName { get; set; }
		public IdentExpr FunctionName { get; set; }
		public SqlExprList FileGroupList { get; set; }
		public string All { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"CREATE PARTITION SCHEME {SchemeName}");
			sb.AppendLine($"AS PARTITION {FunctionName}");
			if (string.IsNullOrEmpty(All))
			{
				sb.Append($"{All} ");
			}
			sb.AppendLine($"TO (");
			sb.AppendLine($"{FileGroupList}");
			sb.Append($")");
			return sb.ToString();
		}
	}
}