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

			var tableName = parser.ConsumeObjectIdOrVariable();

			if(parser.Scanner.TryConsumeAny(out var execSpan, SqlToken.Exec, SqlToken.Execute))
			{
				var execExpr =	parser.PrefixParse(execSpan) as SqlCodeExpr;
				return new InsertIntoFromSqlCodeExpr
				{
					Table = tableName,
					SelectFromExpr = execExpr,
				};
			}

			var columns = GetColumnsList(parser);

			var outputList = parser.GetOutputListExpr();
			var outputInto = parser.GetOutputIntoExpr();

			if (parser.Scanner.TryConsume(SqlToken.Select, out var selectToken))
			{
				var selectExpr = new SelectParselet().Parse(selectToken, parser) as SqlCodeExpr;
				return new InsertIntoFromSqlCodeExpr
				{
					Table = tableName,
					ColumnsList = columns,
					OutputList = outputList,
					OutputIntoExpr = outputInto,
					SelectFromExpr = selectExpr,
				};
			}


			parser.Scanner.Consume(SqlToken.Values);
			var valuesList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				parser.Scanner.Consume(SqlToken.LParen);
				var values = new List<SqlCodeExpr>();
				do
				{
					var expr = parser.ParseExpIgnoreComment();
					values.Add(expr);
				} while (parser.Scanner.Match(SqlToken.Comma));
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

		private static List<string> GetColumnsList(IParser parser)
		{
			var columns = new List<string>();
			if (parser.Scanner.Match(SqlToken.LParen))
			{
				columns = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, SqlToken.Identifier, SqlToken.SqlIdentifier)
					.ToList();
				parser.Scanner.Consume(SqlToken.RParen);
			}

			return columns;
		}
	}
}