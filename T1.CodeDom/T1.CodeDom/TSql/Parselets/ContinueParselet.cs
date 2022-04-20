using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class ContinueParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			return new ContinueSqlCodeExpr();
		}
	}
}