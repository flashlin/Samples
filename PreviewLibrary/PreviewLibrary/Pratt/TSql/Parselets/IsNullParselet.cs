using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class IsNullParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);
			var checkExpression = parser.ParseExpIgnoreComment();
			parser.Scanner.Consume(SqlToken.Comma);
			var replacementValue = parser.ParseExpIgnoreComment();
			parser.Scanner.Consume(SqlToken.RParen);
			return new IsNullSqlCodeExpr
			{
				CheckExpr = checkExpression,
				ReplacementValue = replacementValue
			};
		}
	}
}
