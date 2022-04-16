using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class DeleteParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Match(SqlToken.From);

			var table = parser.ConsumeObjectId();

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.Match(SqlToken.Where))
			{
				whereExpr = parser.ParseExp() as SqlCodeExpr;
			}

			return new DeleteSqlCodeExpr
			{
				Table = table,
				WhereExpr = whereExpr
			};
		}
	}
}