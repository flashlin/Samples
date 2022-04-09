using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class SelectParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var columns = new List<SqlCodeExpr>();
			do
			{
				columns.Add(parser.ParseExp() as SqlCodeExpr);
			} while (parser.Match(SqlToken.Comma));

			return new SelectSqlCodeExpr
			{

			};
		}
	}
}
