using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ExistsParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);
			var innerExpr = parser.ParseExp() as SqlCodeExpr;
			parser.Scanner.ConsumeAny(SqlToken.RParen);
			return new ExistsSqlCodeExpr
			{
				InnerExpr = innerExpr,
			};
		}
	}
}