using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DistinctParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var value = parser.ParseExpIgnoreComment();
			return new DistinctSqlCodeExpr()
			{
				Value = value,
			};
		}
	}
}
