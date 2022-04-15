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
			parser.Scanner.Consume(SqlToken.LParen);

			var dataType = parser.ConsumeDataType();

			parser.Scanner.Consume(SqlToken.Comma);

			var expr = parser.ParseExpIgnoreComment();

			//var style = string.Empty;

			parser.Scanner.Consume(SqlToken.RParen);


			var funcName = new ObjectIdSqlCodeExpr
			{
				ObjectName = "CONVERT"
			};

			var parametersList = new List<SqlCodeExpr>();
			parametersList.Add(dataType);
			parametersList.Add(expr);

			return new FuncSqlCodeExpr
			{
				Name = funcName,
				Parameters = parametersList
			};
		}
	}
}