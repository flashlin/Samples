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
			SqlCodeExpr inputExpr = null;

			if (!parser.Scanner.IsToken(SqlToken.When))
			{
				inputExpr = parser.ParseExpIgnoreComment() as SqlCodeExpr;
			}

			var whenList = ParseWhenList(parser);

			SqlCodeExpr elseExpr = null;
			if (parser.Scanner.Match(SqlToken.Else))
			{
				elseExpr = parser.ParseExp() as SqlCodeExpr;
			}

			parser.Scanner.Consume(SqlToken.End);

			return new CaseSqlCodeExpr
			{
				InputExpr = inputExpr,
				WhenList = whenList,
				ElseExpr = elseExpr,
			};
		}

		private static List<SqlCodeExpr> ParseWhenList(IParser parser)
		{
			var whenList = new List<SqlCodeExpr>();
			do
			{
				if (!parser.Scanner.Match(SqlToken.When))
				{
					break;
				}
				var whenConditionExpr = parser.ParseExpIgnoreComment();
				parser.Scanner.Consume(SqlToken.Then);
				var thenExpr = parser.ParseExpIgnoreComment();
				whenList.Add(new WhenSqlCodeExpr
				{
					ConditionExpr = whenConditionExpr,
					ThenExpr = thenExpr
				});
			} while (true);
			return whenList;
		}
	}
}