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

			//parser.TryConsumeObjectId(out var table);
			var table = parser.ConsumeTableName();

			var withOptions = parser.ParseWithOption();

			parser.Scanner.Consume(SqlToken.Set);

			var setList = ParseSetItemList(parser);

			var outputList = parser.ParseOutputListExpr();

			var intoExpr = ParseIntoExpr(parser);

			var fromTableList = new List<SqlCodeExpr>();
			if (parser.Scanner.Match(SqlToken.From))
			{
				fromTableList = parser.ParseFromSourceList();
			}

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.Match(SqlToken.Where))
			{
				whereExpr = parser.ParseExpIgnoreComment();
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
			
			var intoTable = parser.ConsumeTableName(int.MaxValue);

			var columnList = new List<SqlCodeExpr>();
			if (parser.MatchToken(SqlToken.LParen))
			{
				do
				{
					var column = parser.ConsumeObjectId(true);
					columnList.Add(column);
				} while (parser.Scanner.Match(SqlToken.Comma));
				parser.ConsumeToken(SqlToken.RParen);
			}

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
				//var column = parser.ConsumeObjectIdOrVariable(int.MaxValue);
				var column = parser.ParseMeetObjectId();
				if (column == null)
				{
					ThrowHelper.ThrowParseException(parser, "Expect column");
				}

				var oper = parser.ConsumeTokenString();

				var valueExpr = parser.ParseExpIgnoreComment();
				valueExpr = parser.ParseLRParenExpr(valueExpr);

				setList.Add(new AssignSqlCodeExpr
				{
					Left = column,
					Oper = oper,
					Right = valueExpr
				});
			} while (parser.Scanner.Match(SqlToken.Comma));
			return setList;
		}
	}
}
