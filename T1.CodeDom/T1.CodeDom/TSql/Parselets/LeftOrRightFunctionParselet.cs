using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class LeftOrRightFunctionParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var name = parser.Scanner.GetSpanString(token);
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
					ObjectName = name
				},
				Parameters = parametersList
			};
		}
	}
}
