using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class GroupParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var expr = parser.ParseExpIgnoreComment();

			if( parser.IsToken(SqlToken.LParen))
			{
				expr = parser.ParseLRParenExpr(expr);
			}

			parser.Scanner.Consume(SqlToken.RParen);
			return new GroupSqlCodeExpr
			{
				InnerExpr = expr
			};
		}
	}
}
