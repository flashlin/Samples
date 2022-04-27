using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class PivotParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var pivotStr = parser.Scanner.GetSpanString(token);
			
			parser.Scanner.Consume(SqlToken.LParen);

			var aggregated = parser.ParseExpIgnoreComment();
			parser.Scanner.Consume(SqlToken.FOR);
			
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
				Token = pivotStr.ToUpper(),
				Aggregated = aggregated,
				Column = column,
				PivotedColumns = pivotedColumnsList,
				AliasName = aliasName
			};
		}
	}
}
