using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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