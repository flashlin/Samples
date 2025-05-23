﻿using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class TempTableParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);
			var tempTableExpr = new TempTableSqlCodeExpr
			{
				Name = tokenStr
			};

			if (parser.MatchToken(SqlToken.Dot))
			{
				var tableSourceExpr = new TableSourceSqlCodeExpr
				{
					Table = tempTableExpr,
					Column = parser.ConsumeTokenStringAny(SqlToken.Identifier)
				};
				return tableSourceExpr;
			}

			return tempTableExpr;
		}
	}
}
