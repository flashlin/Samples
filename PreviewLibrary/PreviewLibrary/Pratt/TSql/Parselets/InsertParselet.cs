using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class InsertParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var intoStr = string.Empty;
			if (parser.Scanner.TryConsume(SqlToken.Into, out var intoToken))
			{
				intoStr = parser.Scanner.GetSpanString(intoToken);
			}

			var tableName = parser.Scanner.ConsumeObjectId();

			parser.Scanner.Consume(SqlToken.LParen);
			var columns = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, SqlToken.Identifier, SqlToken.SqlIdentifier)
				.ToList();
			parser.Scanner.Consume(SqlToken.RParen);

			parser.Scanner.Consume(SqlToken.Values);

			var valuesList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				parser.Scanner.Consume(SqlToken.LParen);
				var values = new List<SqlCodeExpr>();
				do
				{
					var expr = parser.ParseExp() as SqlCodeExpr;
					values.Add(expr);
				} while (parser.Scanner.Match(SqlToken.Comma));
				//var values = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
				//{
				//	return parser.ParseExp() as SqlCodeExpr;
				//});
				parser.Scanner.Consume(SqlToken.RParen);

				return new ExprListSqlCodeExpr
				{
					Items = values.ToList()
				};
			}).ToList();

			return new InsertSqlCodeExpr
			{
				IntoStr = intoStr,
				TableName = tableName,
				Columns = columns,
				ValuesList = valuesList
			};
		}
	}
}