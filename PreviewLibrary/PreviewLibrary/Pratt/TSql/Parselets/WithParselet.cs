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
			var table = parser.Scanner.ConsumeObjectId();

			var columns = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				var column = parser.Scanner.ConsumeObjectId();
				columns.Add(column);
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			parser.Scanner.Consume(SqlToken.As);
			parser.Scanner.Consume(SqlToken.LParen);
			var innerExpr = parser.GetParseExpIgnoreCommentFunc()();
			parser.Scanner.Consume(SqlToken.RParen);

			return new WithSqlCodeExpr
			{
				Table = table,
				Columns = columns,
				InnerExpr = innerExpr
			};
		}
	}
}