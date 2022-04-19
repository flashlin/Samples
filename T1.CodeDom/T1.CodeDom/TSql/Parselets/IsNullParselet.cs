using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
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
