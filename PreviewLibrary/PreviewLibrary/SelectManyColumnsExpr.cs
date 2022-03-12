using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class TableExpr : SqlExpr
	{
		public IdentExpr Name { get; set; }
		public string AliasName { get; set; }
		public WithOptionsExpr WithOptions { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{Name}");
			if (!string.IsNullOrEmpty(AliasName))
			{
				sb.Append($" {AliasName}");
			}
			if( WithOptions != null)
			{
				sb.Append($" {WithOptions}");
			}
			return sb.ToString();
		}
	}
}