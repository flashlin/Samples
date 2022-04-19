using PreviewLibrary.Pratt.TSql.Expressions;
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
				//FromTable = fromTable,
				//JoinSelectList = joinSelectList,
				FromTableList = fromTableList,
				OutputList = outputList,
				WhereExpr = whereExpr
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
