﻿using System.Text;
using T1.SqlDomParser;

namespace T1.SqlDom.Expressions
{
	public class SelectExpr : SqlExpr
	{
		public List<SqlExpr> Columns { get; set; } = new();
		public List<TableExpr> Tables { get; set; } = new();
		public SqlExpr WhereClause { get; set; } = Empty;
		public override string ToSqlString()
		{
			var sb = new StringBuilder();
			sb.Append("SELECT ");
			sb.Append(Columns.ToSqlString(","));
			if (Tables.Count > 0)
			{
				sb.Append(" FROM ");
				sb.Append(Tables.ToSqlString(","));
			}
			if (WhereClause != Empty)
			{
				sb.Append(" WHERE " + WhereClause.ToSqlString());
			}
			return sb.ToString();
		}
	}
}
