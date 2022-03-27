using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class InsertExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public SqlExprList Fields { get; set; }
		public SqlExprList ValuesList { get; set; }
		public bool IntoToggle { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append("INSERT");
			if(IntoToggle)
			{
				sb.Append(" INTO");
			}
			sb.Append($" {Table}");
			if(Fields != null)
			{
				sb.Append($"({Fields})");
			}
			sb.Append($" VALUES( {ValuesList} )");
			return sb.ToString();
		}
	}
}