using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
			parser.Scanner.ConsumeAny(SqlToken.RParen);
			
			var funcNameExpr = new ObjectIdSqlCodeExpr
			{
				ObjectName = funcName
			};
			
			return new FuncSqlCodeExpr
			{
				Name = funcNameExpr,
				Parameters = arguments
			};
		}
	}
}