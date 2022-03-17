using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary
{
	public class UpdateExpr : SqlExpr
	{
		public SqlExprList Fields { get; set; }
		public SqlExpr WhereExpr { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine("UPDATE SET ");
			foreach (var field in Fields.Items)
			{
				if( field != Fields.Items.First() )
				{
					sb.Append(",");
				}
				sb.AppendLine($"{field}");
			}
			if( WhereExpr != null)
			{
				sb.Append($"WHERE {WhereExpr}");
			}
			return sb.ToString();
		}
	}
}