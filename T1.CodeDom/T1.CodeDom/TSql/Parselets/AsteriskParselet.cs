using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class AsteriskParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			return new AsteriskSqlCodeExpr();
		}
	}
}
