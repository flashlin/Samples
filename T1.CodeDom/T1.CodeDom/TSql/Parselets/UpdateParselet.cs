using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class UpdateParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var topCount = parser.ParseTopCountExpr();

			//var table = parser.ConsumeObjectId();
			parser.TryConsumeObjectId(out var table);

			var withOptions = parser.ParseWithOptions();

			parser.Scanner.Consume(SqlToken.Set);

			var setList = ParseSetItemList(parser);

			var fromTableList = new List<SqlCodeExpr>();
			if (parser.Scanner.Match(SqlToken.From))
			{
				fromTableList = parser.ParseFromSourceList();
			}

			var outputList = parser.GetOutputListExpr();

			var intoExpr = ParseIntoExpr(parser);

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.Match(SqlToken.Where))
			{
				whereExpr = parser.ParseExp() as SqlCodeExpr;
			}

			return new UpdateSqlCodeExpr
			{
				TopCount = topCount,
				Table = table,
				WithOptions = withOptions,
				SetColumnsList = setList,
				FromTableList = fromTableList,
				OutputList = outputList,
				IntoExpr = intoExpr,
				WhereExpr = whereExpr
			};
		}

		private IntoSqlCodeExpr ParseIntoExpr(IParser parser)
		{
			if (!parser.Scanner.Match(SqlToken.Into))
			{
				return null;
			}
			var intoTable = parser.ConsumeTableName();

			var columnList = new List<SqlCodeExpr>();
			parser.ConsumeToken(SqlToken.LParen);
			do{
				var column = parser.ConsumeObjectId();
				columnList.Add(column);
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.ConsumeToken(SqlToken.RParen);

			return new IntoSqlCodeExpr
			{
				IntoTable = intoTable,
				ColumnList = columnList
			};
		}

		private static List<SqlCodeExpr> ParseSetItemList(IParser parser)
		{
			var setList = new List<SqlCodeExpr>();
			do
			{
				var column = parser.ConsumeObjectIdOrVariable(int.MaxValue);
				parser.Scanner.Consume(SqlToken.Equal);
				var valueExpr = parser.ParseExpIgnoreComment();
				setList.Add(new AssignSqlCodeExpr
				{
					Left = column,
					Right = valueExpr
				});
			} while (parser.Scanner.Match(SqlToken.Comma));
			return setList;
		}
	}
}
