using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class LeftFunctionParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);
			var characterExpression = parser.ParseExpIgnoreComment();
			parser.Scanner.Consume(SqlToken.Comma);
			var integerExpression = parser.ParseExpIgnoreComment();
			parser.Scanner.Consume(SqlToken.RParen);

			var parametersList = new List<SqlCodeExpr>();
			parametersList.Add(characterExpression);
			parametersList.Add(integerExpression);

			return new FuncSqlCodeExpr
			{
				Name = new ObjectIdSqlCodeExpr 
				{ 
					ObjectName = "LEFT"
				},
				Parameters = parametersList
			};
		}
	}
}
