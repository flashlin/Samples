﻿using PreviewLibrary.Exceptions;
using PreviewLibrary.Expressions;
using System.Text;

namespace PreviewLibrary.RecursiveParser
{
	public class IntoExpr : SqlExpr
	{
		public IdentExpr Table { get; set; }
		public SqlExprList Columns { get; set; }

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append($"INTO {Table}");
			sb.Append("(");
			sb.Append($"{Columns}");
			sb.Append(")");
			return sb.ToString();
		}
	}
}