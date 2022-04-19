using System;
using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class IfParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var conditionExpr = parser.ParseExp();

			var bodyList = ParseBeginBodyOrBody(parser);

			var elseIfList = ParseElseIfList(parser);

			var elseExpr = new List<SqlCodeExpr>();
			if (parser.Scanner.TryConsume(SqlToken.Else, out var elseSpan))
			{
				elseExpr = ParseBeginBodyOrBody(parser);
			}

			return new IfSqlCodeExpr
			{
				Condition = conditionExpr as SqlCodeExpr,
				Body = bodyList,
				ElseIfList = elseIfList,
				ElseExpr = elseExpr
			};
		}

		private List<SqlCodeExpr> ParseElseIfList(IParser parser)
		{
			var elseIfExprList = new List<SqlCodeExpr>();
			do
			{
				if (!parser.Scanner.IsTokenList(SqlToken.Else, SqlToken.If))
				{
					break;
				}
				elseIfExprList.Add(ParseElseIf(parser));
			} while (true);
			return elseIfExprList;
		}

		private static List<SqlCodeExpr> ParseBeginBodyOrBody(IParser parser)
		{
			var bodyList = new List<SqlCodeExpr>();
			if (parser.Scanner.IsToken(SqlToken.Begin))
			{
				bodyList = parser.ConsumeBeginBody();
			}
			else
			{
				var body = parser.ParseExpIgnoreComment();
				bodyList.Add(body);
			}
			return bodyList;
		}

		private SqlCodeExpr ParseElseIf(IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Else);
			parser.Scanner.Consume(SqlToken.If);

			var conditionExpr = parser.ParseExpIgnoreComment();
			var body = ParseBeginBodyOrBody(parser);

			return new ElseIfSqlCodeExpr
			{
				ConditionExpr = conditionExpr,
				Body = body,
			};
		}
	}
}