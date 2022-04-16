using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class RankParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);
			parser.Scanner.Consume(SqlToken.RParen);
			parser.Scanner.Consume(SqlToken.Over);


			parser.Scanner.Consume(SqlToken.LParen);
			//PARTITION BY i.LocationID

			parser.Scanner.Consume(SqlToken.Order);
			parser.Scanner.Consume(SqlToken.By);

			var sortExprList = new List<SqlCodeExpr>();
			do
			{
				var name = parser.ConsumeObjectId();
				parser.Scanner.TryConsumeAny(out var sortTokenSpan, SqlToken.Asc, SqlToken.Desc);
				var sortToken = parser.Scanner.GetSpanString(sortTokenSpan);
				sortExprList.Add(new SortSqlCodeExpr
				{
					Name = name,
					SortToken = sortToken
				});
			} while (parser.Scanner.Match(SqlToken.Comma));

			parser.Scanner.Consume(SqlToken.RParen);

			return new RankSqlCodeExpr
			{
				OrderByClause = sortExprList
			};
		}
	}
}
