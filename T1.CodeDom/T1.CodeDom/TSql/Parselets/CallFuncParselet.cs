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

			var parameterList = parser.ParseParameterList($"Function '{funcName}'", _minParameterCount, _maxParameterCount);

			return new FuncSqlCodeExpr
			{
				Name = funcNameExpr,
				Parameters = parameterList
			};
		}
	}
}