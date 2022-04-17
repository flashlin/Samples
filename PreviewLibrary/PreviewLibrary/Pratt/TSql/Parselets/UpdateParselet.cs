using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class UpdateParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var topCount = parser.ParseTopCount();

			var table = parser.ConsumeObjectId();

			var withOptions = parser.ParseWithOptions();

			parser.Scanner.Consume(SqlToken.Set);

			var setList = ParseSetItemList(parser);

			SqlCodeExpr fromTable = null;
			if (parser.Scanner.Match(SqlToken.From))
			{
				fromTable = parser.PrefixParseAny(int.MaxValue, SqlToken.Identifier);
				parser.TryConsumeAliasName(out var aliasName);
				var fromTableWithOptions = parser.ParseWithOptions();
				fromTable = new FromSourceSqlCodeExpr
				{
					Left = fromTable,
					AliasName = aliasName,
					Options = fromTableWithOptions
				};
			}

			var joinSelectList = parser.GetJoinSelectList();


			var outputList = new List<SqlCodeExpr>();
			if (parser.Scanner.Match(SqlToken.Output))
			{
				do
				{
					var actionName = parser.Scanner.ConsumeStringAny(SqlToken.Deleted, SqlToken.Inserted);
					parser.Scanner.Consume(SqlToken.Dot);
					var columnName = parser.ConsumeObjectId();
					outputList.Add(new OutputSqlCodeExpr
					{
						OutputActionName = actionName,
						ColumnName = columnName,
					});
				} while (parser.Scanner.Match(SqlToken.Comma));
			}

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
				FromTable = fromTable,
				JoinSelectList = joinSelectList,
				OutputList = outputList,
				WhereExpr = whereExpr
			};
		}

		private static List<SqlCodeExpr> ParseSetItemList(IParser parser)
		{
			var setList = new List<SqlCodeExpr>();
			do
			{
				var column = parser.ConsumeObjectId();
				parser.Scanner.Consume(SqlToken.Equal);
				var expression = parser.ParseExp();
				var valueExpr = expression as SqlCodeExpr;
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
