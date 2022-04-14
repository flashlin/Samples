using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class CaseParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{


			var whenList = new List<SqlCodeExpr>();
			do
			{
				if (!parser.Scanner.Match(SqlToken.When))
				{
					break;
				}
				var whenConditionExpr = parser.ParseExp() as SqlCodeExpr;
				parser.Scanner.Consume(SqlToken.Then);
				var thenExpr = parser.ParseExp() as SqlCodeExpr;
				whenList.Add(new WhenSqlCodeExpr
				{
					ConditionExpr = whenConditionExpr,
					ThenExpr = thenExpr
				});
			} while (true);


			SqlCodeExpr elseExpr = null;
			if (parser.Scanner.Match(SqlToken.Else))
			{
				elseExpr = parser.ParseExp() as SqlCodeExpr;
			}

			parser.Scanner.Consume(SqlToken.End);

			return new CaseSqlCodeExpr
			{
				WhenList = whenList,
				ElseExpr = elseExpr,
			};
		}
	}
}