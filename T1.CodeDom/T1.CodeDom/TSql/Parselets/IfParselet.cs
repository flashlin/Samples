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

			var body = parser.ParseExpIgnoreComment();

			var elseIfList = ParseElseIfList(parser);

			SqlCodeExpr elseExpr = null;
			if (parser.Scanner.TryConsume(SqlToken.Else, out var elseSpan))
			{
				elseExpr = parser.ParseExpIgnoreComment();
			}

			return new IfSqlCodeExpr
			{
				Condition = conditionExpr as SqlCodeExpr,
				Body = body,
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

		//private static List<SqlCodeExpr> ParseBeginBodyOrBody(IParser parser)
		//{
		//	var bodyList = new List<SqlCodeExpr>();
		//	if (parser.Scanner.IsToken(SqlToken.Begin))
		//	{
		//		bodyList = parser.ConsumeBeginBody();
		//	}
		//	else
		//	{
		//		var body = parser.ParseExpIgnoreComment();
		//		bodyList.Add(body);
		//	}
		//	return bodyList;
		//}

		private SqlCodeExpr ParseElseIf(IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Else);

			if(parser.TryConsumeToken(out var beginSpan, SqlToken.Begin))
			{
				var elseExpr = ParseBegin(beginSpan, parser);
				return new ElseIfSqlCodeExpr
				{
					//Body = new List<SqlCodeExpr>() { elseExpr }
					Body = elseExpr
				};
			}

			parser.Scanner.Consume(SqlToken.If);
			var conditionExpr = parser.ParseExpIgnoreComment();
			//var body = ParseBeginBodyOrBody(parser);
			var body = parser.ParseExpIgnoreComment();

			return new ElseIfSqlCodeExpr
			{
				ConditionExpr = conditionExpr,
				Body = body,
			};
		}

		private SqlCodeExpr ParseBegin(TextSpan beginSpan, IParser parser)
		{			
			return parser.PrefixParse(beginSpan) as SqlCodeExpr;
		}
	}
}