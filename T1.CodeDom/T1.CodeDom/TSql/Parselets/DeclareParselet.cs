using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
						Oper = "=",
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