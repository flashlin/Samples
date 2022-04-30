using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class CallFuncParselet : IPrefixParselet
	{
		private readonly int _minParameterCount;
		private readonly int _maxParameterCount;

		public CallFuncParselet(int minParameterCount = 0, int maxParameterCount = int.MaxValue)
		{
			this._minParameterCount = minParameterCount;
			this._maxParameterCount = maxParameterCount;
		}

		public IExpression Parse(TextSpan token, IParser parser)
		{
			var funcName = parser.Scanner.GetSpanString(token);
			if (token.Type != TokenType.Identifier.ToString())
			{
				funcName = funcName.ToUpper();
			}
			var funcNameExpr = new ObjectIdSqlCodeExpr
			{
				ObjectName = funcName
			};

			if (!parser.IsToken(SqlToken.LParen))
			{
				token.Type = SqlToken.Identifier.ToString();
				return parser.PrefixParse(token);
			}

			var parameterList = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				if (parser.IsToken(SqlToken.RParen))
				{
					break;
				}
				if (parameterList.Count >= _maxParameterCount)
				{
					ThrowHelper.ThrowParseException(parser, $"Too many parameters for function '{funcName}', max is {_maxParameterCount}.");
				}
				var parameter = parser.ParseExpIgnoreComment();
				parameterList.Add(parameter);
			} while (parser.MatchToken(SqlToken.Comma));
			parser.ConsumeToken(SqlToken.RParen);

			if (parameterList.Count < this._minParameterCount)
			{
				ThrowHelper.ThrowParseException(parser, $"Function '{funcName}' requires at least {_minParameterCount} parameters.");
			}

			return new FuncSqlCodeExpr
			{
				Name = funcNameExpr,
				Parameters = parameterList
			};
		}
	}
}