using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ObjectIdParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.SetOffset(token.Offset - 1);
			var identExpr = parser.ConsumeObjectId();

			if (parser.Scanner.TryConsume(SqlToken.LParen, out _))
			{
				return Call(identExpr, parser);
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
