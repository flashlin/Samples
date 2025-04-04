﻿using PreviewLibrary.Exceptions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class InnerJoinExpr : SqlExpr
	{
		public JoinType JoinType { get; set; }
		public string OuterToken { get; set; }
		public SqlExpr Table { get; set; }
		public IdentExpr AliasName { get; set; }
		public SqlExpr OnFilter { get; set; }
		public WithOptionsExpr WithOptions { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"{JoinType}");
			if (!string.IsNullOrEmpty(OuterToken))
			{
				sb.Append($" {OuterToken}");
			}
			sb.Append($" JOIN {Table}");
			if (AliasName != null)
			{
				sb.Append($" AS {AliasName}");
			}
			if (WithOptions != null)
			{
				sb.Append($" {WithOptions}");
			}
			sb.Append($" ON {OnFilter}");
			return sb.ToString();
		}
	}
}