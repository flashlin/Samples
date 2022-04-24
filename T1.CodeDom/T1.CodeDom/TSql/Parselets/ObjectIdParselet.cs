using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class ObjectIdParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var startIndex = parser.Scanner.GetOffset();
			parser.Scanner.SetOffset(token.Offset - 1);
			//var identExpr = parser.ConsumeObjectId();

			if (!parser.TryConsumeObjectId(out var identExpr))
			{
				parser.Scanner.SetOffset(startIndex);
				ThrowHelper.ThrowParseException(parser, "Expected object id");
				//parser.Scanner.SetOffset(startIndex);
				//return new CallFuncParselet().Parse(token, parser);
			}

			if( parser.MatchToken(SqlToken.LParen))
			{
				var parameterList = new List<SqlCodeExpr>();
				do
				{
					if (parser.IsToken(SqlToken.RParen))
					{
						break;
					}
					var p = parser.ParseExpIgnoreComment();
					parameterList.Add(p);
				} while (parser.MatchToken(SqlToken.Comma));
				parser.ConsumeToken(SqlToken.RParen);

				identExpr = new FuncSqlCodeExpr
				{
					Name = identExpr,
					Parameters = parameterList
				};
			}

			return identExpr;
		}

		private static IExpression Call(SqlCodeExpr identExpr, IParser parser)
		{
			var parametersList = new List<SqlCodeExpr>();
			if (!parser.Scanner.TryConsume(SqlToken.RParen, out _))
			{
				do
				{
					var parameter = parser.ParseExp();
					parametersList.Add(parameter as SqlCodeExpr);
				} while (parser.Scanner.TryConsume(SqlToken.Comma, out _));
				parser.Scanner.Consume(SqlToken.RParen);
			}
			return new FuncSqlCodeExpr
			{
				Name = identExpr,
				Parameters = parametersList
			};
		}
	}
}
