using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class PivotParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);

			var aggregated = parser.ParseExpIgnoreComment();
			parser.Scanner.Consume(SqlToken.For);
			
			var column = parser.ConsumeObjectId();
			
			parser.Scanner.Consume(SqlToken.In);

			parser.Scanner.Consume(SqlToken.LParen);
			var pivotedColumnsList = new List<SqlCodeExpr>();
			do
			{
				var pivotedColumn = parser.ParseExpIgnoreComment();
				pivotedColumnsList.Add(pivotedColumn as SqlCodeExpr);
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			parser.Scanner.Consume(SqlToken.RParen);
			parser.Scanner.Match(SqlToken.As);

			var aliasName = parser.ConsumeObjectId();

			return new PivotSqlCodeExpr
			{
				Aggregated = aggregated,
				Column = column,
				PivotedColumns = pivotedColumnsList,
				AliasName = aliasName
			};
		}
	}
}
