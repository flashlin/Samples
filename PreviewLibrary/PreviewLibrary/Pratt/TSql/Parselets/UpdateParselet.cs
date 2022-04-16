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

			var table = parser.Scanner.ConsumeObjectId();

			var withOptions = parser.ParseWithOptions();

			parser.Scanner.Consume(SqlToken.Set);

			var setList = new List<SqlCodeExpr>();
			do
			{
				var column = parser.Scanner.ConsumeObjectId();
				parser.Scanner.Consume(SqlToken.Equal);
				var expression = parser.ParseExp();
				var valueExpr = expression as SqlCodeExpr;
				setList.Add(new AssignSqlCodeExpr
				{
					Left = column,
					Right = valueExpr
				});
			} while (parser.Scanner.Match(SqlToken.Comma));

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
				WhereExpr = whereExpr
			};
		}
	}
}
