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
			var table = parser.Scanner.ConsumeObjectId();

			parser.Scanner.Consume(SqlToken.Set);

			var setList = new List<SqlCodeExpr>();
			do
			{
				var column = parser.Scanner.ConsumeObjectId();
				parser.Scanner.Consume(SqlToken.Equal);
				var valueExpr = parser.ParseExp() as SqlCodeExpr;
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
				Table = table,
				SetColumnsList = setList,
				WhereExpr = whereExpr
			};
		}
	}
}
