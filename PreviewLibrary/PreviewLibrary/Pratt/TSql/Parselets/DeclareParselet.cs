using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class DeclareParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var variableList = new List<SqlCodeExpr>();
			do
			{
				if (!parser.TryConsume(SqlToken.Variable, out var varName))
				{
					break;
				}

				parser.Scanner.Match(SqlToken.As);
				var dataTypeExpr = parser.ConsumeDataType();

				SqlCodeExpr variableDataType = new DeclareSqlCodeExpr
				{
					Name = varName as SqlCodeExpr,
					DataType = dataTypeExpr,
				};

				if (parser.Scanner.Match(SqlToken.Equal))
				{
					var valueExpr = parser.ParseExpIgnoreComment();
					variableDataType = new AssignSqlCodeExpr
					{
						Left = variableDataType,
						Right = valueExpr
					};
				}
				variableList.Add(variableDataType);
			} while (parser.Scanner.Match(SqlToken.Comma));

			return new ExprListSqlCodeExpr
			{
				IsComma = false,
				Items = variableList
			};
		}
	}
}