using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class NotParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
         var right = parser.ParseExp() as SqlCodeExpr;
         return new NotSqlCodeExpr
			{
            Right = right,
			};
		}
	}
}