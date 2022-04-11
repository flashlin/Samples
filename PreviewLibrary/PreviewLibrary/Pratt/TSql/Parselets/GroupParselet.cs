using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class GroupParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var expr = parser.ParseExp() as SqlCodeExpr;
			parser.Scanner.ConsumeTokenType(SqlToken.RParen);
			return new GroupSqlCodeExpr
			{
				InnerExpr = expr
			};
		}
	}
}
