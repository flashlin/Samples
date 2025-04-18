﻿using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DeleteParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var topCount = parser.ParseTopCountExpr();

			parser.Scanner.Match(SqlToken.From);

			var table = parser.ConsumeTableName();

			var withOptions = parser.ParseWithOption();

			var outputList = parser.ParseOutputListExpr();
			var intoTableExpr = parser.ParseInto_TableExpr();
			
			var outputList1 = parser.ParseOutputListExpr();

			var fromSourceList = new List<SqlCodeExpr>();
			if(parser.Scanner.Match(SqlToken.From))
			{
				fromSourceList = parser.ParseFromSourceList();
			}

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.Match(SqlToken.Where))
			{
				whereExpr = parser.ParseExp() as SqlCodeExpr;
			}

			var optionExpr = parser.ParseOptionExpr();

			return new DeleteSqlCodeExpr
			{
				TopExpr = topCount,
				Table = table,
				WithOptions = withOptions,
				OutputList = outputList,
				IntoTableExpr = intoTableExpr,
				OutputList1 = outputList1,
				FromSourceList = fromSourceList,
				WhereExpr = whereExpr,
				OptionExpr = optionExpr
			};
		}
	}
}