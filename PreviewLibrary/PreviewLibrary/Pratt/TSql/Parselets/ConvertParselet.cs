using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ConvertParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var parametersList = new List<SqlCodeExpr>();

			parser.Scanner.Consume(SqlToken.LParen);

			var dataType = parser.ConsumeDataType();
			parametersList.Add(dataType);

			parser.Scanner.Consume(SqlToken.Comma);

			var expr = parser.ParseExpIgnoreComment();
			parametersList.Add(expr);

			if(parser.Scanner.Match(SqlToken.Comma))
			{
				var styleSpan = parser.Scanner.Consume();
				var style = parser.PrefixParse(styleSpan) as SqlCodeExpr;
				parametersList.Add(style);
			}

			parser.Scanner.Consume(SqlToken.RParen);
			var funcName = new ObjectIdSqlCodeExpr
			{
				ObjectName = "CONVERT"
			};

			return new FuncSqlCodeExpr
			{
				Name = funcName,
				Parameters = parametersList
			};
		}
	}
}