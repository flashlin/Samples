using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary
{
	public class GrantToExpr : SqlExpr
	{
		public string Permission { get; set; }
		public SqlExprList ToObjectIds { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"GRANT {Permission}");
			sb.Append($" {ToObjectIds}");
			return sb.ToString();
		}
	}
}