using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class FuncParselet : IPrefixParselet
	{
		private readonly int _maxParameterCount;

		public FuncParselet(int maxParameterCount)
		{
			this._maxParameterCount = maxParameterCount;
		}

		public IExpression Parse(TextSpan token, IParser parser)
		{
			var funcName = parser.Scanner.GetSpanString(token);

			parser.Scanner.Consume(SqlToken.LParen);

			var arguments = new List<SqlCodeExpr>();
			do
			{
				if (arguments.Count >= _maxParameterCount)
				{
					ThrowHelper.ThrowParseException(parser, $"Expect Max Parameter Count={_maxParameterCount} for '{funcName}' Function.");
				}
				var innerExpr = parser.ParseExpIgnoreComment();
				arguments.Add(innerExpr);
			} while (parser.Scanner.Match(SqlToken.Comma));

			var funcNameExpr = new ObjectIdSqlCodeExpr
			{
				ObjectName = funcName
			};

			parser.Scanner.ConsumeAny(SqlToken.RParen);
			return new FuncSqlCodeExpr
			{
				Name = funcNameExpr,
				Parameters = arguments
			};
		}
	}
}