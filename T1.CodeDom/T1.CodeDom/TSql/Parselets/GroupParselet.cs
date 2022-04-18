using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class GroupParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var expr = parser.ParseExp() as SqlCodeExpr;
			parser.Scanner.Consume(SqlToken.RParen);
			return new GroupSqlCodeExpr
			{
				InnerExpr = expr
			};
		}
	}
}
