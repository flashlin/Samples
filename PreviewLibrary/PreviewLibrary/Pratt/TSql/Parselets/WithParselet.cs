using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class WithParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var items = new List<WithItemSqlCodeExpr>();
			do
			{
				var table = parser.ConsumeObjectId();

				var columns = new List<SqlCodeExpr>();
				if (parser.Scanner.Match(SqlToken.LParen))
				{
					do
					{
						var column = parser.ConsumeObjectId();
						columns.Add(column);
					} while (parser.Scanner.Match(SqlToken.Comma));
					parser.Scanner.Consume(SqlToken.RParen);
				}

				parser.Scanner.Consume(SqlToken.As);
				parser.Scanner.Consume(SqlToken.LParen);
				var innerExpr = parser.ParseExpIgnoreComment();
				parser.Scanner.Consume(SqlToken.RParen);

				items.Add(new WithItemSqlCodeExpr
				{
					Table = table,
					Columns = columns,
					InnerExpr = innerExpr
				});
			} while (parser.Scanner.Match(SqlToken.Comma));

			return new WithSqlCodeExpr
			{
				Items = items
			};
		}
	}
}